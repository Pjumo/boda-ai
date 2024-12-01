def modify_pose_json(data, key_to_check):
    updated_data = []
    for item in data:
        if item.get(key_to_check) == 1 or item.get(key_to_check) == 2:
            continue
        elif item.get(key_to_check) == 0:
            item[key_to_check] = 7
        elif item.get(key_to_check) == 3:
            item[key_to_check] = 8
        elif item.get(key_to_check) == 4:
            item[key_to_check] = 9
        updated_data.append(item)
    return updated_data
