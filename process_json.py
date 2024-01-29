import json

new_dic = {}
with open('demo/validation_interactive_tfexample.tfrecord-00009-of-00150.json') as f:
    data = json.load(f)
    for each_scenario in data:
        data_dic = data[each_scenario]
        del data_dic['raw']
        new_dic[each_scenario] = data_dic
        for each_key in data_dic:
            print(each_key)
        # print(data[each_key])
with open('test.json', 'w') as fp:
    json.dump(new_dic, fp)
