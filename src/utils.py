

def delete_keys_from_dict(dict, keys):
    return {k: v for k, v in dict.items() if k not in keys}