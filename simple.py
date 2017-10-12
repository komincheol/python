language = ['python','per','c','java']

for lang in language:
    if lang in ['python','perl']:
        print("%6s need interpreter" % lang)
        elif lang in('c', 'java'):
            print("%6s need compiler" % lang)
            else: 
                print("should not reach here")
