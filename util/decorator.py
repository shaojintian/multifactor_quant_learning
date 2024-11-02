import functools

def print_variable_shapes(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 执行被装饰的函数
        result = func(*args, **kwargs)
        
        # 获取局部变量
        local_vars = {k: v for k, v in locals().items() if k != 'result'}
        
        # 打印每个局部变量的名称和形状
        print("Variable shapes inside the function:")
        for var_name, var_value in local_vars.items():
            if hasattr(var_value, 'shape'):
                print(f"{var_name}: {var_value.shape}")
            else:
                print(f"{var_name}: Not a numpy array or similar")
        
        return result
    return wrapper