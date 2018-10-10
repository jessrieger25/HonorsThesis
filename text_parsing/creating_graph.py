class CreateGraph:

    def within_range_graph(self, relational_dict, keyword, range):

        with open(os.path.abspath("./g_star_graphs/within_range_" + keyword + '_' + str(range) + '.txt'), 'w') as g_star:
            g_star.write('new graph\n')

            graph_count = []
            vertices = []
            edges = []

            g_star.write('add vertex ' + keyword.strip() + ' with attributes(size=' + str(1000) + ')\n')
            vertices.append(keyword)
            for relation in relational_dict:
                print(relation)
                edge = []
                for relation_key, ind in relation.items():
                    print(relation_key)
                    if relation_key not in vertices:
                        g_star.write('add vertex ' + relation_key + '\n')
                        vertices.append(relation_key)
                    edge.append(relation_key)
                if edge not in edges:
                    edges.append(edge)
                    g_star.write(
                        'add edge ' + edge[0] + ' - ' + edge[1] + '\n')

    def average_distances(self, relational_dict, keyword):

        with open(os.path.abspath("./g_star_graphs/average_dist_" + keyword + '.txt'), 'w') as g_star:
            g_star.write('new graph\n')
            g_star.write('add vertex ' + keyword.strip() + ' with attributes(size=' + str(1000) + ')\n')

            for word, count in relational_dict.items():

                g_star.write('add vertex ' + word.strip() + ' with attributes(size=' + str(count).strip() + ')\n')