def add_condition_combined(ad, condition_keys):
    if "conditions_combined" not in ad.obs:
        if len(condition_keys) > 1:
            ad.obs["conditions_combined"] = ad.obs[condition_keys].apply(
                lambda x: "_".join(x), axis=1
            )
        else:
            ad.obs["conditions_combined"] = ad.obs[condition_keys]
    return ad