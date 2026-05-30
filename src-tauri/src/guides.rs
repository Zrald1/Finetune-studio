//! AMD ROCm fine-tuning guide registry.
//!
//! Maps a student HF repo ID to the matching Unsloth notebook in `guide amd/`
//! and exposes that notebook's recipe (LoRA rank, alpha, learning rate, cutoff
//! length, recommended method) so the pipeline can silently apply validated
//! defaults at run start.
//!
//! Source notebooks (in `C:\Users\Zrald\Fine Tune Model\guide amd\`):
//!   - `AMD-Llama3.3_(70B)_A100-Conversational.ipynb`
//!   - `AMD-Mistral_v0.3_(7B)-Alpaca.ipynb`
//!   - `AMD-Qwen3_(14B)-Reasoning-Conversational.ipynb`
//!   - `Qwen3_(32B)_A100-Reasoning-Conversational.ipynb` (CUDA variant of the above)
//!   - `Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb` (GRPO)
//!   - `gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb` (GRPO)
//!
//! Matching is intentionally permissive (case-insensitive substring on the
//! repo ID); unrecognised models return `None` and the user's existing
//! LoraConfig values are used unchanged.

use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct GuideRecipe {
    pub family: &'static str,
    pub label: &'static str,
    pub notebook: &'static str,
    /// LLaMA-Factory chat template this family ships with. Currently
    /// informational — `llamafactory::pick_template` already does its own
    /// repo-id-based detection, so this is here for future use / docs.
    #[allow(dead_code)]
    pub template: &'static str,
    pub lora_r: u32,
    pub lora_alpha: u32,
    pub lora_dropout: f32,
    pub cutoff_len: u32,
    pub learning_rate: f32,
    pub recommended_method: &'static str,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchedGuideInfo {
    pub family: String,
    pub label: String,
    pub notebook: String,
    pub recommended_method: String,
}

impl From<&'static GuideRecipe> for MatchedGuideInfo {
    fn from(g: &'static GuideRecipe) -> Self {
        MatchedGuideInfo {
            family: g.family.to_string(),
            label: g.label.to_string(),
            notebook: g.notebook.to_string(),
            recommended_method: g.recommended_method.to_string(),
        }
    }
}

// Recipes lifted from each notebook's `get_peft_model` + `SFTConfig` /
// `GRPOConfig` cells. Order matters: more specific families first so e.g.
// `gpt-oss` matches before any accidental `gpt` substring elsewhere.
static GUIDES: &[GuideRecipe] = &[
    GuideRecipe {
        family: "gpt-oss",
        label: "AMD gpt-oss (20B) GRPO RL",
        notebook: "gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb",
        template: "default",
        lora_r: 4,
        lora_alpha: 8,
        lora_dropout: 0.0,
        cutoff_len: 768,
        learning_rate: 5e-5,
        recommended_method: "grpo",
    },
    GuideRecipe {
        family: "gemma-4",
        label: "AMD Gemma-4 GRPO RL",
        notebook: "Gemma4_(E2B)_Reinforcement_Learning_Sudoku_Game.ipynb",
        template: "gemma",
        lora_r: 32,
        lora_alpha: 64,
        lora_dropout: 0.0,
        cutoff_len: 4096,
        learning_rate: 5e-5,
        recommended_method: "grpo",
    },
    GuideRecipe {
        family: "qwen3",
        label: "AMD Qwen3 Reasoning Conversational",
        notebook: "AMD-Qwen3_(14B)-Reasoning-Conversational.ipynb",
        template: "qwen3",
        lora_r: 32,
        lora_alpha: 32,
        lora_dropout: 0.0,
        cutoff_len: 2048,
        learning_rate: 2e-4,
        recommended_method: "unsloth",
    },
    GuideRecipe {
        family: "llama-3",
        label: "AMD Llama-3.x Conversational",
        notebook: "AMD-Llama3.3_(70B)_A100-Conversational.ipynb",
        template: "llama3",
        lora_r: 16,
        lora_alpha: 16,
        lora_dropout: 0.0,
        cutoff_len: 2048,
        learning_rate: 2e-4,
        recommended_method: "unsloth",
    },
    GuideRecipe {
        family: "mistral-v0.3",
        label: "AMD Mistral v0.3 Alpaca",
        notebook: "AMD-Mistral_v0.3_(7B)-Alpaca.ipynb",
        template: "mistral",
        lora_r: 16,
        lora_alpha: 16,
        lora_dropout: 0.0,
        cutoff_len: 2048,
        learning_rate: 2e-4,
        recommended_method: "unsloth",
    },
];

/// Match an HF repo ID to a recipe. Case-insensitive substring search on a
/// normalised version of the repo ID (slashes replaced with spaces so e.g.
/// `Qwen/Qwen3-14B` and `unsloth/Qwen3-14B-unsloth-bnb-4bit` both hit `qwen3`).
///
/// Returns `None` for models outside the covered families (Qwen2.5, Phi,
/// DeepSeek, etc.) so the user's existing settings stay untouched.
pub fn match_guide(student_model: &str) -> Option<&'static GuideRecipe> {
    let lower = student_model.to_lowercase();

    // gpt-oss is its own family — match first so it never falls into a
    // generic `gpt` bucket later.
    if lower.contains("gpt-oss") || lower.contains("gpt_oss") {
        return GUIDES.iter().find(|g| g.family == "gpt-oss");
    }
    // Gemma-4 only; older Gemma-2/3 are not covered by the GRPO notebook so
    // we deliberately don't match them.
    if lower.contains("gemma-4") || lower.contains("gemma4") {
        return GUIDES.iter().find(|g| g.family == "gemma-4");
    }
    if lower.contains("qwen3") {
        return GUIDES.iter().find(|g| g.family == "qwen3");
    }
    // Llama-3 family (3.1 / 3.2 / 3.3). Exclude Llama-2/Llama (no AMD guide).
    if lower.contains("llama-3") || lower.contains("llama3") {
        return GUIDES.iter().find(|g| g.family == "llama-3");
    }
    // Mistral v0.3 specifically — earlier Mistral versions have a different
    // tokenizer and aren't in the AMD recipe.
    if lower.contains("mistral") && (lower.contains("v0.3") || lower.contains("v03")) {
        return GUIDES.iter().find(|g| g.family == "mistral-v0.3");
    }
    None
}

/// Apply the matched guide's settings to a LoraConfig **only for fields that
/// still hold the schema default**. A user who explicitly tuned `cutoff_len`
/// or `learning_rate` keeps their value.
///
/// The "default" detection mirrors the `Default` impl on LoraConfig and the
/// schema defaults from the TypeScript side (see `DEFAULT_LORA` in
/// `FineTune/src/types.ts`): zero or a small placeholder value.
pub fn apply_guide_defaults(lora: &mut crate::runs::LoraConfig, g: &GuideRecipe) {
    // r=0 is invalid (LoRA requires r>=1) so we treat 0 or 8 as default and
    // override; any other value is treated as user intent.
    if lora.r == 0 || lora.r == 8 {
        lora.r = g.lora_r;
    }
    if lora.alpha == 0 || lora.alpha == 8 || lora.alpha == 16 {
        // 16 is the legacy DEFAULT_LORA value — safe to override.
        if lora.alpha != g.lora_alpha {
            lora.alpha = g.lora_alpha;
        }
    }
    // Dropout default is 0.0; leave alone if non-zero.
    if lora.dropout == 0.0 {
        lora.dropout = g.lora_dropout;
    }
    if lora.cutoff_len == 0 || lora.cutoff_len == 1024 || lora.cutoff_len == 2048 {
        lora.cutoff_len = g.cutoff_len;
    }
    // Common legacy defaults: 1e-4, 2e-4, 5e-5. Override these; leave anything
    // else as user-set.
    if (lora.learning_rate - 1e-4).abs() < f32::EPSILON
        || (lora.learning_rate - 2e-4).abs() < f32::EPSILON
        || (lora.learning_rate - 5e-5).abs() < f32::EPSILON
        || lora.learning_rate == 0.0
    {
        lora.learning_rate = g.learning_rate;
    }
    // Method: only auto-switch if the user is on plain `lora` (the schema
    // default) AND the guide recommends something different. Don't clobber an
    // explicit method choice.
    let cur = lora.method.trim().to_lowercase();
    if (cur.is_empty() || cur == "lora") && g.recommended_method != "lora" {
        lora.method = g.recommended_method.to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runs::LoraConfig;

    fn default_lora() -> LoraConfig {
        LoraConfig {
            method: "lora".to_string(),
            custom_method_name: String::new(),
            custom_commands: vec![],
            unsloth_backend: "cuda".to_string(),
            r: 8,
            alpha: 16,
            dropout: 0.0,
            learning_rate: 2e-4,
            epochs: 1.0,
            batch_size: 2,
            gradient_accumulation: 4,
            cutoff_len: 2048,
            save_steps: 100,
        }
    }

    #[test]
    fn matches_qwen3_family() {
        let g = match_guide("unsloth/Qwen3-14B-unsloth-bnb-4bit").unwrap();
        assert_eq!(g.family, "qwen3");
        assert_eq!(g.lora_r, 32);
        assert_eq!(g.lora_alpha, 32);
        assert_eq!(g.template, "qwen3");
    }

    #[test]
    fn matches_qwen3_official_repo() {
        let g = match_guide("Qwen/Qwen3-32B").unwrap();
        assert_eq!(g.family, "qwen3");
    }

    #[test]
    fn skips_qwen2_5() {
        // Qwen2.5 is the legacy default — no AMD guide covers it, so we must
        // return None and let the user's existing settings stand.
        assert!(match_guide("Qwen/Qwen2.5-7B-Instruct").is_none());
    }

    #[test]
    fn matches_gpt_oss_grpo() {
        let g = match_guide("unsloth/gpt-oss-20b-BF16").unwrap();
        assert_eq!(g.family, "gpt-oss");
        assert_eq!(g.recommended_method, "grpo");
    }

    #[test]
    fn matches_llama_3() {
        let g = match_guide("meta-llama/Meta-Llama-3.1-70B-Instruct").unwrap();
        assert_eq!(g.family, "llama-3");
    }

    #[test]
    fn matches_llama_3_lowercase() {
        let g = match_guide("unsloth/llama-3.2-3b-instruct").unwrap();
        assert_eq!(g.family, "llama-3");
    }

    #[test]
    fn skips_llama_2() {
        assert!(match_guide("meta-llama/Llama-2-7b").is_none());
    }

    #[test]
    fn matches_mistral_v03_only() {
        let g = match_guide("unsloth/mistral-7b-instruct-v0.3-bnb-4bit").unwrap();
        assert_eq!(g.family, "mistral-v0.3");
        // Earlier Mistral versions should not match.
        assert!(match_guide("mistralai/Mistral-7B-Instruct-v0.2").is_none());
    }

    #[test]
    fn matches_gemma_4_only() {
        let g = match_guide("unsloth/gemma-4-E2B-it").unwrap();
        assert_eq!(g.family, "gemma-4");
        assert_eq!(g.recommended_method, "grpo");
        assert!(match_guide("google/gemma-2-9b-it").is_none());
    }

    #[test]
    fn apply_overrides_default_rank_and_alpha_for_qwen3() {
        let g = match_guide("Qwen/Qwen3-14B").unwrap();
        let mut lora = default_lora();
        apply_guide_defaults(&mut lora, g);
        assert_eq!(lora.r, 32);
        assert_eq!(lora.alpha, 32);
        assert_eq!(lora.cutoff_len, 2048); // already matched
    }

    #[test]
    fn apply_preserves_user_cutoff_override() {
        let g = match_guide("Qwen/Qwen3-14B").unwrap();
        let mut lora = default_lora();
        lora.cutoff_len = 4096; // user changed it
        apply_guide_defaults(&mut lora, g);
        assert_eq!(lora.cutoff_len, 4096, "user override must be preserved");
    }

    #[test]
    fn apply_preserves_user_method_override() {
        let g = match_guide("Qwen/Qwen3-14B").unwrap();
        let mut lora = default_lora();
        lora.method = "qlora".to_string(); // user picked QLoRA explicitly
        apply_guide_defaults(&mut lora, g);
        assert_eq!(lora.method, "qlora", "user method choice must be preserved");
    }

    #[test]
    fn apply_switches_default_lora_to_grpo_for_gpt_oss() {
        let g = match_guide("unsloth/gpt-oss-20b-BF16").unwrap();
        let mut lora = default_lora();
        apply_guide_defaults(&mut lora, g);
        assert_eq!(lora.method, "grpo");
        assert_eq!(lora.cutoff_len, 768);
    }
}
