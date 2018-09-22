from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


eng_stopwords = stopwords.words('english')
unmodified_string = ""
with open("/Users/Jess/PycharmProjects/HonorsThesis/TimeMachine1.txt", "r") as time:
    for line in time.readlines():
        unmodified_string += line.replace('\n', " ")

print(unmodified_string)


custom_sent_tokenizer = PunktSentenceTokenizer(unmodified_string)
tokenized = custom_sent_tokenizer.tokenize(unmodified_string)

print(tokenized)
words = []
for i in tokenized:
    words += nltk.word_tokenize(i)
    print(words)

parsed_string = []
for i in words:
    temp = i.lower()
    print("tis is temp " + temp)
    if temp not in eng_stopwords and temp not in ['but', '\'', 'and', 'us', 'you', '?', ',', '.', 'at', 'the']:
        parsed_string.append(temp)
print(parsed_string)

# vertices = {}
# edges = []
# with open('/Users/Jess/PycharmProjects/HonorsThesis/tester.txt', 'a') as g_star:
#     g_star.write('new graph\n')
#
#     for ind in range(1, len(parsed_string)-1):
#         for index in range(ind-1, ind+2):
#             if parsed_string[index] not in vertices:
#                 g_star.write('add vertex ' + parsed_string[index] + '\n')
#
#                 vertices[parsed_string[index]] = 1
#             else:
#                 vertices[parsed_string[index]] = vertices[parsed_string[index]] + 1
#         first_edge = 'add edge ' + parsed_string[ind-1] + ' - ' + parsed_string[ind]
#         if first_edge not in edges:
#             g_star.write(first_edge + '\n')
#         second_edge = 'add edge ' + parsed_string[ind] + ' - ' + parsed_string[ind+1]
#
#         if second_edge not in edges:
#             g_star.write(second_edge + '\n')
#
#     for key, value in vertices.items():
#         g_star.write('update ' + key + ' with attributes(size=' + str(value) + ')\n')


# POS section
tagged_list = []
for i in parsed_string:
    print(words)
    # print(nltk.pos_tag(i))
    # tagged = nltk.pos_tag(unmodified_string)
    # tagged_list.append(tagged)
# text = nltk.word_tokenize(parsed_string)
tagged = nltk.pos_tag(parsed_string)

pos_vertices = {}
pos_edges = []
print(tagged)
with open('pos_listing_full2.txt', 'w') as pos:
    pos.write('new graph\n')

    for ind in range(1, len(tagged) - 1):
        for index in range(ind - 1, ind + 2):
            if tagged[index][0] not in pos_vertices:
                pos.write('add vertex ' + tagged[index][0] + '\n')

                pos_vertices[tagged[index][0]] = 1
            else:
                pos_vertices[tagged[index][0]] = pos_vertices[tagged[index][0]] + 1

        first_edge = 'add edge ' + tagged[ind - 1][0] + ' - ' + tagged[ind][0]

        if first_edge not in pos_edges:
            pos.write(first_edge + '\n')
            pos_edges.append(first_edge)

        second_edge = 'add edge ' + tagged[ind][0] + ' - ' + tagged[ind + 1][0]

        if second_edge not in pos_edges:
            pos.write(second_edge + '\n')
            pos_edges.append(second_edge)

    for key, value in pos_vertices.items():
        pos.write('update ' + key + ' with attributes(size=' + str(value) + ')\n')
        print(key)
        print(nltk.pos_tag([key]))
        temp = nltk.pos_tag(key)[0]

        temp = temp[1]

        pos.write('update ' + key + ' with attributes(pos=' + temp + ')\n')
















