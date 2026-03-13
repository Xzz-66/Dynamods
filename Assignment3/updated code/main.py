# from src.config import Config
# from src.batch import run_ensemble


# def main():
#     cfg = Config()
#     out_dir = "runs"
#     seeds = list(range(1, 26))

#     run_ensemble("mild", cfg.sI_mild, seeds, out_dir, cfg)
#     run_ensemble("strong", cfg.sI_strong, seeds, out_dir, cfg)


# if __name__ == "__main__":
#     main()

# lazy debug
from src.config import Config
from src.runner import run_single


def main():
    cfg = Config()

    
    cfg.n_macro = 20
    cfg.MCS_per_macro = 40
    cfg.pde_substeps = 6
    cfg.snapshot_stride = 20
    cfg.save_snapshots = True

    summary = run_single(
        condition_name="mild",
        sI_value=cfg.sI_strong,
        seed=1,
        out_dir="runs",
        cfg=cfg,
    )

    print("Finished single debug run.")
    print(summary)


if __name__ == "__main__":
    main()