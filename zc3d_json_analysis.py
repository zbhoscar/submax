import data_io.basepy as basepy

test_eval_result_json_path = '/absolute/tensorflow_models/190601162431/190601162431.ckpt-7847_anoma_json'
test_eval_result_json_list = basepy.get_1tier_file_path_list(test_eval_result_json_path, suffix='.json')

