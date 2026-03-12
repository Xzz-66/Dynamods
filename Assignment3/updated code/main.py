from src.config import Config
from src.batch import run_ensemble


def main():
    cfg = Config()
    out_dir = "runs"
    seeds = list(range(1, 26))

    run_ensemble("mild", cfg.sI_mild, seeds, out_dir, cfg)
    run_ensemble("strong", cfg.sI_strong, seeds, out_dir, cfg)


if __name__ == "__main__":
    main()


