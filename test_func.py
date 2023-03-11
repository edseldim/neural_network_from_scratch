def replace_nested_list(nested_list, flat_list, flat_list_index = 0):
    for i in range(len(nested_list)):
        if isinstance(nested_list[i], list):
            flat_list_index = replace_nested_list(nested_list[i], flat_list, flat_list_index)
        else:
            nested_list[i] = flat_list[flat_list_index]
            flat_list_index += 1
    return flat_list_index

nested = [[],[0,1],[[2,3]],[5]]
flat = [4,5,6,7,8]
replace_nested_list(nested, flat)
print(nested)