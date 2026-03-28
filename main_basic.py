from lcip_app.launchers import train_new_model_basic


if __name__ == "__main__":
    raise SystemExit(
        train_new_model_basic(
            dataset_name="mnist",
            P_name="tsne",
            Pinv_name="lcip",
            clf=True,
            GRID=100,
        )
    )
