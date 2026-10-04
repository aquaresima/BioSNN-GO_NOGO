import numpy as np
import pandas as pd
import os
import copy


def build_network(
    rsnn,
    datapath,
    area_list=["ALM", "AC"],
    with_video=True,
    hidden_propability=0.0,
    spread_sessions=False,
):
    """
    Build the network based on the data collected from the dataset and based on the array
    rsnn.area_index and rsnn.excitatory_index that specifies the area  and neurotransmitter type
    of each neuron.
    Sample without replacement from the list of the neurons from the sessions
    that have video behavior when with_video True, else sample from any possible session.
    Also, choose p_exc of the neurons to be excitatory. The excitation is defined as
    Peak to Peak time in the spike template is more than 0.56ms.
    """
    clusterdf = pd.read_csv(os.path.join(datapath, "cluster_information"))
    clusterdfwhole = clusterdf.copy()  # for statistics for hidden neurons
    if with_video:
        clusterdf = clusterdf[clusterdf["with_video"] == with_video]
    clusterdf = clusterdf[clusterdf["area"].isin(area_list)]
    # keep neurons with less than 40 Hz
    if "Vahid" in datapath:
        clusterdf = clusterdf[clusterdf.firing_rate < 40]

    n_neurons = rsnn.n_units
    # init of the output arrays
    firing_rate = np.zeros(n_neurons)
    neuron_index = np.zeros(n_neurons)
    session = np.zeros(n_neurons).astype(str)
    areas = np.zeros(n_neurons).astype(str)

    # order sessions based on the number of neurons they have
    # this prioritize sessions with more neurons
    sessions_uniq = clusterdf.session.value_counts().index.values
    neurons_per_session = [
        clusterdf[clusterdf.session == sessions_uniq[i]].shape[0]
        for i in range(len(sessions_uniq))
    ]
    areas_per_session = [
        clusterdf[clusterdf.session == sessions_uniq[i]].area.unique().shape[0]
        for i in range(len(sessions_uniq))
    ]
    # remove sessions with less than 40 neurons and 1 area or less than 80 neurons and 2 areas
    sessions_uniq = [
        sessions_uniq[i]
        for i in range(len(sessions_uniq))
        if not (neurons_per_session[i] < 60 and areas_per_session[i] > 1)
        and not (neurons_per_session[i] < 30)
    ]

    sess_exc = [
        np.array(
            [
                sess
                for sess in sessions_uniq
                if area in clusterdf.area[clusterdf.session == sess].values
            ]
        )
        for area in area_list
    ]
    sess_inh = copy.deepcopy(sess_exc)

    planned = None
    if spread_sessions:
        # "collage" over all sessions of an area: the model neurons of every (area, E/I) group are split over the
        # sessions in proportion to how many neurons of that group each session has (largest remainder), instead
        # of filling the biggest session first. Every session then constrains the model with its own trials.
        planned = np.empty(n_neurons, dtype=object)
        for area_id, area_name in enumerate(area_list):
            for exc_flag in (True, False):
                slots = [
                    i
                    for i in range(n_neurons)
                    if int(rsnn.area_index[i]) == area_id
                    and bool(rsnn.excitatory_index[i]) == exc_flag
                ]
                avail = clusterdf[
                    (clusterdf["area"] == area_name)
                    & (clusterdf["excitatory"] == exc_flag)
                    & (clusterdf["session"].isin(sessions_uniq))
                ].session.value_counts()
                if len(slots) == 0:
                    continue
                assert avail.sum() >= len(slots), "not enough neurons in the dataset"
                share = avail.values / avail.values.sum() * len(slots)
                counts = np.minimum(np.floor(share).astype(int), avail.values)
                rest = len(slots) - counts.sum()
                for j in np.argsort(-(share - np.floor(share)))[:rest]:
                    counts[j] += 1
                names = np.repeat(avail.index.values, counts)
                planned[slots] = names[np.random.permutation(len(names))]
    # in this loop we make the matching from recordings to simulations
    for i in range(n_neurons):
        exc = rsnn.excitatory_index[i].numpy()
        area_id = rsnn.area_index[i]
        area = area_list[area_id]
        cur_session = sess_exc[area_id] if exc else sess_inh[area_id]
        if planned is not None:
            cur_session = np.array([planned[i]])

        possible_ids = clusterdf[
            (clusterdf["area"] == area)
            & (clusterdf["excitatory"] == exc)
            & (clusterdf["session"] == cur_session[0])
        ]

        # making sure that we use all neurons from a session and then move to the next session
        while possible_ids.shape[0] == 0 and area != "Nope":
            cur_session = cur_session[1:]
            possible_ids = clusterdf[
                (clusterdf["area"] == area)
                & (clusterdf["excitatory"] == exc)
                & (clusterdf["session"] == cur_session[0])
            ]

        if possible_ids.shape[0] == 0 and area != "Nope":
            if hidden_propability == 0:
                assert False, "not enough neurons in the dataset"
            else:
                print("filling with hidden")

        # in the first case of the if statement we check if we add a hidden neuron
        # we add hidden neurons, if we have hidden_probability > 0 or the area is a Nope area
        flag_hidden = np.random.rand() < hidden_propability
        if flag_hidden or area == "Nope":
            neuron_index[i] = -1  # default for hidden neurons
            session[i] = ""
            areas[i] = area
            # if neuron is hidden, use firing rate the firing rate of random neuron from the dataset
            possible_ids = clusterdfwhole[
                (clusterdfwhole["area"] == area) & (clusterdfwhole["excitatory"] == exc)
            ]
            # use the firing rate of a random neuron from same group (area/exc-inh)
            index = np.random.choice(clusterdfwhole.index)
            firing_rate[i] = clusterdfwhole.firing_rate[
                clusterdfwhole.index == index
            ].values
        else:
            # from the possible neurons pick one
            index = np.random.choice(possible_ids.index)
            neuron_index[i] = clusterdf.cluster_index[clusterdf.index == index].values
            firing_rate[i] = clusterdf.firing_rate[clusterdf.index == index].values
            session[i] = clusterdf.session[clusterdf.index == index].values[0]
            areas[i] = clusterdf.area[clusterdf.index == index].values[0]
            clusterdf = clusterdf[clusterdf.index != index]
    return neuron_index, firing_rate, session, areas

