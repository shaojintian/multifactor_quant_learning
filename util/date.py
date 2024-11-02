import datetime

def date_to_utc_milliseconds(date_str):
    # 将字符串转换为 datetime 对象
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    
    # 将 datetime 对象转换为 UTC
    dt_utc = dt.replace(tzinfo=datetime.timezone.utc)
    
    # 获取 UTC 的时间戳（秒）
    timestamp_s = dt_utc.timestamp()
    
    # 转换为毫秒
    timestamp_ms = int(timestamp_s * 1000)
    
    return timestamp_ms

# 示例调用
date_str = "2021-10-01"
utc_milliseconds = date_to_utc_milliseconds(date_str)

#print(f"{date_str} 的 UTC 毫秒时间戳是: {utc_milliseconds}")