import os
import util.logger as logger

def setup_logging_directory(out_dir, env_name, method, run_id):
    method = method.split("/")
    log_dir = [out_dir, env_name, *method, run_id]
    log_dir = os.path.sep.join(log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    eval_dir = os.path.join(log_dir, "eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    
    return log_dir


def escape_forward_slash(s:str):
    return s.replace("/", ":")


def record_param(params, values, group_name, step):
    log_record = logger.record
    
    for param, value in zip(params, values):
        unit = param.unit
        log_record(f"{group_name}/{param.name} in " \
                   f"{escape_forward_slash(unit)}", value, step)


def record_dict(d, group_name, step):
    log_record = logger.record
    
    for key in list(d):
        log_record(f"{group_name}/{escape_forward_slash(key)}", d[key], step)
