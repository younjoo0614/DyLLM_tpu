import os
from dyllm.dllm import dLLM
from dyllm.sampling_params import SamplingParams
from transformers import AutoTokenizer

def _detect_tpu_tp_size(default: int = 1) -> int:
    if os.environ.get("DYLLM_TP_SIZE"):
        return max(1, int(os.environ["DYLLM_TP_SIZE"]))
    # Do not probe torch_xla devices here: eager discovery can initialize PJRT
    # and keep /dev/vfio/* busy before DLLMEngine spawns worker ranks.
    # For multi-rank TPU, set DYLLM_TP_SIZE explicitly.
    return default


def _pjrt_device_kind() -> str:
    # Accept both env spellings and normalize to torch-xla's PJRT_DEVICE.
    device = (os.environ.get("PJRT_DEVICE") or os.environ.get("PJRT_DEVICES") or "").strip().upper()
    if device and "PJRT_DEVICE" not in os.environ:
        os.environ["PJRT_DEVICE"] = device
    return device


def main():

    #path = os.path.expanduser("/home/yunju/data/model/Dream-v0-Instruct-7B")
    path = os.path.expanduser("/home/yunju/data/model/LLaDA-8B-Instruct")

    on_tpu = _pjrt_device_kind() == "TPU"
    tp_size = _detect_tpu_tp_size(default=1) if on_tpu else 1

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    dllm = dLLM(
        path,
        threshold=0.99,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
        backend="xla" if on_tpu else "auto",
        runtime_device="xla" if on_tpu else "auto",
    )

    sampling_params = SamplingParams(
        temperature=None,
        max_new_tokens=256,
        steps=256,
        num_full_steps=4,
        block_size=32,
        ignore_eos=True,
        algorithm="confidence",
    )

    prompts = [
        "Describe the water cycle in detail.",        
    ]

    templated = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]

    outputs = dllm.generate(templated, sampling_params)

    for p, out in zip(prompts, outputs):
        print("\nPrompt:", repr(p))
        print("Completion:", repr(out["text"]))


if __name__ == "__main__":
    main()

