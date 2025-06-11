# train.py
import os
import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from src.modeling.UNet import UNet
from src.modeling.RUNet import RUNet
from src.data_processing.loader import MyoPSDataset
from metrics import multiclass_dice_loss, dice_LVbloodpool, dice_RVbloodpool, dice_LVmyo, dice_LVmyoedema, dice_LVmyoscars

# 读取配置
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    train_cfg = load_config("configs/train_config.yaml")
    data_cfg = load_config("configs/data_config.yaml")
    model_cfg = load_config("configs/model_config.yaml")

    # 数据加载
    train_dataset = MyoPSDataset(
        split_path=data_cfg["split_json"],
        mode="train",
        batch_size=train_cfg["batch_size"],
        data_dir=data_cfg["train_data_dir"]
    )
    val_dataset = MyoPSDataset(
        split_path=data_cfg["split_json"],
        mode="val",
        batch_size=train_cfg["batch_size"],
        data_dir=data_cfg["val_data_dir"]
    )

    # 模型构建
    input_shape = tuple(model_cfg["input_shape"])
    num_classes = model_cfg["num_classes"]

    if model_cfg.get("model_type", "UNet") == "RUNet":
        model = RUNet(input_shape, num_classes)
    else:
        model = UNet(input_shape, num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(train_cfg["learning_rate"]),
        loss=multiclass_dice_loss,
        metrics=[dice_LVbloodpool, dice_RVbloodpool, dice_LVmyo, dice_LVmyoedema, dice_LVmyoscars]
    )

    # 回调函数
    callbacks = [
        ModelCheckpoint(train_cfg["checkpoint_path"], monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(patience=5, factor=0.5, verbose=1),
        EarlyStopping(patience=10, verbose=1),
        TensorBoard(log_dir=train_cfg["log_dir"])
    ]

    # 训练
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=train_cfg["epochs"],
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()
