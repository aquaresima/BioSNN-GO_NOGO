import torch
import numpy as np
import pandas as pd

from infopath.config import load_training_opt
from infopath.model_loader import load_model_and_optimizer
from infopath.utils.functions import load_data
from infopath.lick_classifier import prepare_classifier

from infopath.opto import opto_effect


LOG_PATH = "log_dir/main/2026_6_5_6_21_35_full"
CHECKPOINT = "best"


def main():

    print(f"Loading model from {LOG_PATH}")

    opt = load_training_opt(LOG_PATH)
    opt.log_path = LOG_PATH
    opt.device = "cuda"

    model = load_model_and_optimizer(
        opt,
        reload=True,
        last_best=CHECKPOINT,
    )[0]

    print("Model loaded")

    (
        train_spikes,
        train_jaw,
        session_info_train,
        test_spikes,
        test_jaw,
        session_info_test,
    ) = load_data(model)

    print("Data loaded")

    filt_jaw_train = model.filter_fun1(train_jaw)
    filt_jaw_test = model.filter_fun1(test_jaw)

    lick_classifier = prepare_classifier(
        model.filter_fun2(filt_jaw_train),
        model.filter_fun2(filt_jaw_test),
        session_info_train,
        session_info_test,
        opt.device,
    )

    model.rsnn.light_neuron = (
        1 - model.rsnn.excitatory_index.long()
    ).bool()

    batch_size = 200
    trials = 200

    model.opt.batch_size = batch_size

    time_vector = np.arange(model.T) * 0.002 - 0.15

    dfs = []

    for seed in range(20):

        print(f"Running seed {seed}")

        torch.manual_seed(seed)

        stims = torch.randint(
            2,
            size=(trials,),
            device=model.opt.device,
        )

        state = model.steady_state()

        input_spikes = model.input_spikes(stims)

        model.rsnn.sample_mem_noise(model.T, trials)

        mem_noise = model.rsnn.mem_noise.clone()

        model.rsnn.sample_trial_noise(trials)

        with torch.no_grad():

            spikes, voltages, jaw, _ = model.step_with_dt(
                input_spikes,
                state,
                mem_noise,
                dt=50,
            )

            filt_model = model.filter_fun1(spikes)

            df = opto_effect(
                time_vector,
                model.filter_fun1(train_spikes),
                filt_model,
                session_info_train,
                model,
                stims,
                input_spikes,
                state,
                mem_noise,
                filt_data_jaw=model.filter_fun1(train_jaw),
                filt_jaw=model.filter_fun1(jaw),
                lick_detector=lick_classifier,
                seed=seed,
                verbose=True,
                power=0.2,
            )

            dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)

    output_file = "opto_results.csv"

    dfs.to_csv(output_file, index=False)

    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
