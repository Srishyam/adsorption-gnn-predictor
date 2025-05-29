''' 
Convert a pkl file into json file
'''
import sys
import os
import pickle
import json
import datetime
from json import JSONEncoder

# subclass JSONEncoder
class DateTimeEncoder(JSONEncoder):
        #Override the default method
        def default(self, obj):
            if isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()

def convert_dict_to_json(file_path):
    with open(file_path, 'rb') as fpkl, open('%s.json' % file_path, 'w') as fjson:
        data = pickle.load(fpkl)
#        print(data[0]['atoms']['atoms'][1]['tag'])
        #json.dump(data, fjson, ensure_ascii=False, sort_keys=True, indent=4)
        json.dump(data, fjson, ensure_ascii=False, sort_keys=True, indent=4, cls=DateTimeEncoder)

def main():
    if sys.argv[1] and os.path.isfile(sys.argv[1]):
        file_path = sys.argv[1]
        print("Processing %s ..." % file_path)
        convert_dict_to_json(file_path)
    else:
        print("Usage: %s abs_file_path" % (__file__))


if __name__ == '__main__':
    main()

''' 
import pickle
import json
import jsonpickle
import codecs
from bson import ObjectId

f_pickle = open('./mat_lists/mat_1.pkl', 'rb')
dict = pickle.load(f_pickle)

#j_string = jsonpickle.encode(dict, unpicklable=False)

class JSONEncoder(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, ObjectId):
			return str(o)
		return json.JSONEncoder.default(self, o)

JSONEncoder().encode(analytics)
with codecs.open('mat_json.json', mode='w', encoding='utf-8') as f_json:
	json.dump(dict, f_json)

'''
