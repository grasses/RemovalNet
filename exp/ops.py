
import numpy as np


def exp11_normalize(results, methods, metrics, defense_method):
    target_models = []
    for model, item in results.items():
        method = model.split("-")[2].split("(")[0]
        if method in methods:
            target_models.append(model)

    # transform data structure
    dists = np.zeros([len(metrics), len(target_models), 10])
    for i, metric in enumerate(metrics):
        for j, model in enumerate(target_models):
            dists[i, j] = results[model][metric]

        if np.mean(dists[i]) < 100:
            print(metric, np.array(dists[i] * 100, dtype=np.int32))
        else:
            print(metric, np.array(dists[i], dtype=np.int32))
        print()
    dists = np.array(dists, dtype=np.float32)

    # data = [metrics, models, 10 ground result]
    v = np.reshape(dists, [len(metrics), -1])
    min_v = np.min(v, axis=1)
    max_v = np.max(v, axis=1)
    dists_nz = np.copy(dists)
    for i, metric in enumerate(metrics):
        for j, model in enumerate(target_models):
            dists_nz[i, j] = (dists[i, j] - min_v[i]) / (max_v[i] - min_v[i] + 1e-6)
            if defense_method in ["ModelDiff"]:
                dists_nz[i, j] = np.ones(len(dists_nz[i, j])) - dists_nz[i, j]

        print(metric, np.array(dists_nz[i] * 100, dtype=np.int32))
        #print(metric, np.array(dists_nz[i] * 100, dtype=np.int32))



    print("-------------> Raw data")
    for j, model in enumerate(target_models):
        print(f"-> Task:{model}")
        for i, metric in enumerate(metrics):
            min_x = round(float(np.min(dists[i, j])), 4)
            max_x = round(float(np.max(dists[i, j])), 4)
            med_x = round(float(np.median(dists[i, j])), 4)
            mean_x = round(float(np.mean(dists[i, j])), 4)
            std_x = round(float(np.std(dists[i, j])), 4)
            print(f"-> metric: {metric} med:{med_x}±{round((max_x-min_x)/2.0, 4)} mean,std=({mean_x},{std_x})")
        print()

    print("-------------> Normalized data")
    for j, model in enumerate(target_models):
        print(f"-> Task:{model}")
        for i, metric in enumerate(metrics):
            min_x = round(float(np.min(dists_nz[i, j])), 4)
            max_x = round(float(np.max(dists_nz[i, j])), 4)
            med_x = round(float(np.median(dists_nz[i, j])), 4)
            mean_x = round(float(np.mean(dists_nz[i, j])), 4)
            std_x = round(float(np.std(dists_nz[i, j])), 4)
            print(f"-> metric: {metric} med:{med_x}±{round((max_x-min_x)/2.0, 4)} mean,std=({mean_x},{std_x})")
        print()

    print(f"-> min:{min_v}, max:{max_v}")

    legends = [f"{defense_method}-{metric}" for metric in metrics]
    xticks = [name.split("-")[-2] for name in target_models]
    ylabel = "Similarity" if defense_method in ["ModelDiff", "IPGuard", "MetaV", "MetaFinger"] else "Distance"
    data = {
        "min_v": min_v,
        "max_v": max_v,
        "xticks": xticks,
        "ylabel": ylabel,
        "legends": legends,
        "metrics": metrics,
        "models": target_models,
        "dists": dists,
        "dists_nz": dists_nz
    }
    return data