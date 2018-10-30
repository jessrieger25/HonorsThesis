import os
from text_parsing.keyword_manager import KeywordManager

class CreateGraph:

    def __init__(self):
        self.keyword_manager = KeywordManager()

    def within_range_graph(self, relational_dict, keyword, range):

        with open(os.path.abspath("../g_star_graphs/within_range_" + keyword + '_' + str(range) + '.txt'), 'w') as g_star:
            g_star.write('new graph\n')

            vertices = []
            edges = []

            g_star.write('add vertex ' + keyword.strip() + ' with attributes(size=' + str(1000) + ')\n')
            vertices.append(keyword)
            for relation in relational_dict:
                edge = []
                for relation_key, ind in relation.items():
                    if relation_key not in vertices:
                        color = KeywordManager().category_colors[KeywordManager().keywords_list[relation_key]]
                        g_star.write('add vertex ' + relation_key + ' with attributes(color=' + color + ')\n')
                        vertices.append(relation_key)
                    edge.append(relation_key)
                if edge not in edges:
                    edges.append(edge)
                    g_star.write(
                        'add edge ' + edge[0] + ' - ' + edge[1] + '\n')

    def average_distances(self, relational_dict, keyword):

        with open(os.path.abspath("../g_star_graphs/average_dist_" + keyword + '.txt'), 'w') as g_star:
            g_star.write('new graph\n')
            g_star.write('add vertex ' + keyword.strip() + ' with attributes(size=' + str(1000) + 'color=black)\n')

            max = 0

            for word, count in relational_dict.items():
                if count > max:
                    max = count

            for word, count in relational_dict.items():
                print("keywords", self.keyword_manager.keywords_list)
                print('word', word + "hi")
                if word != '':
                    print(word)
                    g_star.write('add vertex ' + word.strip() + ' with attributes(size=' + str(abs(count-max)*.8).strip() + ', color=' + KeywordManager().category_colors[KeywordManager().keywords_list[word.strip()]] + ')\n')