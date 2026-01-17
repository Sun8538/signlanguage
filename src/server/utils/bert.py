import torch

from utils.store import Store

# Fix for PyTorch 2.6+ weights_only default change
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


class Bert:
    checker = None
    _initialized = False

    @classmethod
    def _ensure_initialized(cls):
        if not cls._initialized:
            try:
                from neuspell import BertChecker
                cls.checker = BertChecker()
                cls.checker.from_pretrained()
                cls._initialized = True
            except Exception as e:
                print(f"Warning: Could not initialize BertChecker: {e}")
                print("Spell checking will be disabled.")
                cls._initialized = True  # Don't retry

    @classmethod
    def fix(cls):
        print("WEE WOO WEE WOO")

        Store.raw_word = ""

        raw_transcription = " ".join(Store.raw_transcription).lower()
        print(raw_transcription)

        cls._ensure_initialized()
        
        if cls.checker:
            corrected = cls.checker.correct(raw_transcription).strip()
        else:
            # Fallback: no spell correction
            corrected = raw_transcription.strip()
        print(corrected)

        Store.corrected_transcription = corrected.upper().split()
