def alignment(final_factor,ret):
    # 在计算net_values之前对齐数据
    common_index = final_factor.index.intersection(ret.index)
    final_factor = final_factor.loc[common_index]
    ret = ret.loc[common_index]
    return final_factor,ret