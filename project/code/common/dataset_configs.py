"""
Per-dataset configuration registry. One entry per dataset actually selected in
project/data_screening/dataset_screening_report.md. Paths, label column, and
graph-identity columns are taken directly from the screening JSON — not guessed.
"""
import os

DATA_ROOT = r"C:\Users\MMASSAOUDI\Desktop\Data"

DATASETS = {
    "rt_iot2022": {
        "role": "smoke_test",
        "files": [os.path.join(DATA_ROOT, "RT_IOT2022", "RT_IOT2022.csv")],
        "label_col": "Attack_type",
        "drop_cols": ["Unnamed: 0", "no"],  # row-index artifacts seen in this file
        "id_like_cols": ["proto", "service", "id.orig_p", "id.resp_p"],
        # confirmed via direct value_counts() on Attack_type: these 3 are normal-device-traffic
        # categories (IoT cloud telemetry / MQTT / smart-bulb operation), the other 9 are attacks.
        "benign_values": ["Thing_Speak", "MQTT_Publish", "Wipro_bulb"],
        "official_split": False,
    },
    "unsw_nb15": {
        "role": "primary_general",
        "files": {
            "train": os.path.join(DATA_ROOT, "UNSW_NB15", "UNSW_NB15_training-set.csv"),
            "test": os.path.join(DATA_ROOT, "UNSW_NB15", "UNSW_NB15_testing-set.csv"),
        },
        "label_col": "attack_cat",
        "binary_label_col": "label",  # UNSW-NB15 also ships a ready binary label
        "drop_cols": ["id"],
        "id_like_cols": ["proto", "service", "ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm",
                          "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst"],
        "official_split": True,
        "benign_via_binary_label": 0,  # use binary_label_col==0 as benign, not attack_cat string match
    },
    "edge_iiot": {
        "role": "primary_cps_industrial",
        "files": [os.path.join(DATA_ROOT, "Edge-IIoTset", "Edge-IIoTset dataset",
                                "Selected dataset for ML and DL", "ML-EdgeIIoT-dataset.csv")],
        "label_col": "Attack_type",
        "drop_cols": ["frame.time"],  # kept in id_like for graph use, dropped from raw feature matrix separately
        "id_like_cols": ["ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport",
                          "udp.port", "mqtt.protoname", "frame.time"],
        "official_split": False,
        "benign_values": ["Normal"],
        # Normal-traffic device folders in the raw capture (project/data_screening report) —
        # used to construct a REAL benign-distribution-shift split for TTA-GraphRL: hold out some
        # device types from training, test adaptation on the held-out ones. The curated
        # ML-EdgeIIoT-dataset.csv does not retain a device-folder column, so this shift is applied
        # via the raw per-device files in a dedicated loader (see candidates/tta_graphrl.py).
        "normal_device_dirs": ["Distance", "Flame_Sensor", "Heart_Rate", "IR_Receiver", "Modbus",
                                "phValue", "Soil_Moisture", "Sound_Sensor",
                                "Temperature_and_Humidity", "Water_Level"],
    },
    "bot_iot": {
        "role": "primary_graph_temporal",
        "files_glob": os.path.join(DATA_ROOT, "Bot_IoT", "data_*.csv"),
        "label_col": "attack",
        "multiclass_cols": ["category", "subcategory "],  # note trailing space in real header
        "drop_cols": ["pkSeqID"],
        "id_like_cols": ["saddr", "daddr", "sport", "dport", "proto", "smac", "dmac", "stime", "ltime"],
        "official_split": False,
        "benign_via_binary_label": 0,  # attack column: 0=benign, 1=attack
    },
    "cicIot23": {
        "role": "primary_modern_iot",
        "files": {
            "train": os.path.join(DATA_ROOT, "CICIOT23", "CICIOT23", "train", "train.csv"),
            "validation": os.path.join(DATA_ROOT, "CICIOT23", "CICIOT23", "validation", "validation.csv"),
            "test": os.path.join(DATA_ROOT, "CICIOT23", "CICIOT23", "test", "test.csv"),
        },
        "label_col": "label",
        "drop_cols": [],
        "id_like_cols": ["Protocol Type", "IPv"],
        "official_split": True,
        "benign_values": ["BenignTraffic"],
    },
}
