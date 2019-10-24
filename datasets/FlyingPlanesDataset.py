from .TreeDatasetClassification import TreeDatasetClassification


class FlyingPlanesDataset(TreeDatasetClassification):

    def read_labels(self, label_path):
        labels = []
        for t in self.trees:
            subj = t.children[0].children[0]
            if subj.node == 'Ger':
                labels.append(0)    # "is"
            else:
                labels.append(1)    # "are"
        return labels
