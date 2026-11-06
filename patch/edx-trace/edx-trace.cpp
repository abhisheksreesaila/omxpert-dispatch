// Expert Dispatch: per-token MoE routing tracer.
// Runs the prompt file (-f) in chunks of n_ctx and writes, for every token,
// the top-k expert ids of every MoE layer as int32 [T, L, K] to -o.
// Uses cb_eval on "ffn_moe_topk-<il>" (slower than plain inference; fine for tracing).
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

static bool g_hidden = false; // EDX_TRACE_HIDDEN=<file>: dump router inputs attn_post_norm (f16 [T, L, n_embd])
static bool g_probs = false; // EDX_TRACE_PROBS=<file>: also dump router probabilities (f16 [T, L, E])

struct trace_state {
    std::map<int, std::vector<float>> probs; // layer -> [n_tok * E]
    std::map<int, std::vector<float>> hid;   // layer -> [n_tok * n_embd]
    int D = 0;
    int E = 0;
    std::map<int, std::vector<int32_t>> per_layer;  // layer -> ids for this decode, [n_tok * k]
    int k = 0;
    std::vector<uint8_t> buf;
};

static bool edx_cb(struct ggml_tensor * t, bool ask, void * ud) {
    if (g_hidden && strncmp(t->name, "attn_post_norm-", 15) == 0 && t->type == GGML_TYPE_F32) {
        if (ask) return true;
        auto * st = (trace_state *) ud;
        const int il = atoi(t->name + 15);
        const int64_t d = t->ne[0], n = t->ne[1];
        st->D = (int) d;
        auto & v = st->hid[il];
        std::vector<float> row(d);
        for (int64_t j = 0; j < n; ++j) {
            ggml_backend_tensor_get(t, row.data(), j * t->nb[1], d * sizeof(float));
            v.insert(v.end(), row.begin(), row.end());
        }
        return true;
    }
    if (g_probs && strncmp(t->name, "ffn_moe_probs-", 14) == 0 && t->type == GGML_TYPE_F32) {
        if (ask) return true;
        auto * st = (trace_state *) ud;
        const int il = atoi(t->name + 14);
        const int64_t e = t->ne[0], n = t->ne[1];
        st->E = (int) e;
        auto & v = st->probs[il];
        std::vector<float> row(e);
        for (int64_t j = 0; j < n; ++j) {
            ggml_backend_tensor_get(t, row.data(), j * t->nb[1], e * sizeof(float));
            v.insert(v.end(), row.begin(), row.end());
        }
        return true;
    }
    if (strncmp(t->name, "ffn_moe_topk-", 13) != 0) return ask ? false : true;
    if (ask) return true;
    auto * st = (trace_state *) ud;
    const int il = atoi(t->name + 13);
    // selected_experts: I32 [k, n_tokens] (possibly a view; copy rows of ne0)
    const int64_t k = t->ne[0], n = t->ne[1];
    st->k = (int) k;
    if (getenv("EDX_DEBUG")) fprintf(stderr, "EDXDBG %s ne=%lld,%lld,%lld type=%d\n", t->name, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (int) t->type);
    auto & v = st->per_layer[il];
    std::vector<int32_t> row(k);
    for (int64_t j = 0; j < n; ++j) {
        ggml_backend_tensor_get(t, row.data(), j * t->nb[1], k * sizeof(int32_t));
        v.insert(v.end(), row.begin(), row.end());
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.out_file = "trace.bin";
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX)) return 1;
    trace_state st;
    params.cb_eval = edx_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    llama_backend_init();
    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx = init->context();
    if (!model || !ctx) { LOG_ERR("failed to init\n"); return 1; }

    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, true, params.parse_special);
    const int n_ctx = llama_n_ctx(ctx);
    const int n_batch = std::min<int>(params.n_batch, n_ctx);
    FILE * fo = fopen(params.out_file.c_str(), "wb");
    FILE * fp = getenv("EDX_TRACE_PROBS") ? fopen(getenv("EDX_TRACE_PROBS"), "wb") : nullptr;
    g_probs = fp != nullptr;
    FILE * fh = getenv("EDX_TRACE_HIDDEN") ? fopen(getenv("EDX_TRACE_HIDDEN"), "wb") : nullptr;
    g_hidden = fh != nullptr;
    long long T = 0;
    int L = 0;
    for (size_t s = 0; s + 1 < toks.size(); s += n_ctx) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int len = (int) std::min<size_t>(n_ctx, toks.size() - s);
        for (int b = 0; b < len; b += n_batch) {
            const int nb = std::min(n_batch, len - b);
            st.per_layer.clear();
            st.probs.clear();
            st.hid.clear();
            // every token is an output, else the last layer only runs on the final row
            llama_batch batch = llama_batch_init(nb, 0, 1);
            for (int j = 0; j < nb; ++j) {
                batch.token[j] = toks[s + b + j]; batch.pos[j] = b + j;
                batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = 1;
            }
            batch.n_tokens = nb;
            const int rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (rc) { LOG_ERR("decode failed\n"); return 1; }
            if (L == 0) L = (int) st.per_layer.size();
            // write [nb, L, k]
            std::vector<int32_t> out((size_t) nb * L * st.k);
            int li = 0;
            for (auto & [il, v] : st.per_layer) {
                if ((int) v.size() != nb * st.k) { fprintf(stderr, "EDX mismatch layer %d: %zu vs %d\n", il, v.size(), nb * st.k); return 1; }
                for (int j = 0; j < nb; ++j)
                    memcpy(&out[((size_t) j * L + li) * st.k], &v[(size_t) j * st.k], st.k * sizeof(int32_t));
                ++li;
            }
            fwrite(out.data(), sizeof(int32_t), out.size(), fo);
            if (fh && !st.hid.empty()) { // [nb, L, D] as f16
                std::vector<ggml_fp16_t> hv((size_t) nb * st.hid.size() * st.D);
                int lj = 0;
                for (auto & [il, v] : st.hid) {
                    for (int j = 0; j < nb; ++j)
                        for (int d = 0; d < st.D; ++d)
                            hv[((size_t) j * st.hid.size() + lj) * st.D + d] = ggml_fp32_to_fp16(v[(size_t) j * st.D + d]);
                    ++lj;
                }
                fwrite(hv.data(), sizeof(ggml_fp16_t), hv.size(), fh);
            }
            if (fp && !st.probs.empty()) { // [nb, L, E] as f16
                std::vector<ggml_fp16_t> pr((size_t) nb * st.probs.size() * st.E);
                int lj = 0;
                for (auto & [il, v] : st.probs) {
                    for (int j = 0; j < nb; ++j)
                        for (int e = 0; e < st.E; ++e)
                            pr[((size_t) j * st.probs.size() + lj) * st.E + e] = ggml_fp32_to_fp16(v[(size_t) j * st.E + e]);
                    ++lj;
                }
                fwrite(pr.data(), sizeof(ggml_fp16_t), pr.size(), fp);
            }
            T += nb;
        }
        LOG_INF("edx-trace: %lld / %zu tokens\n", T, toks.size());
    }
    fclose(fo);
    if (fp) fclose(fp);
    if (fh) fclose(fh);
    fprintf(stderr, "EDX_TRACE_DONE T=%lld L=%d K=%d\n", T, L, st.k);
    return 0;
}
