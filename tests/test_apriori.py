""" Validate behavior from Apriori algorithms. """
import unittest

from gunter.apriori import create_candidates, support_prune, apriori, generate_rules

class TestApriori(unittest.TestCase):

    def setUp(self):
        
        # list of transactions; each transaction is a list of items
        
        self.dataset = self.load_dataset()

        # set of transactions; each transaction is a list of items
        
        self.D = map(set, self.dataset)

    def load_dataset(self):
        """Loads an example of market basket transactions for testing purposes.

        Returns
        -------
        A list (database) of lists (transactions). Each element of a transaction 
        is an item.
        """
        return [['Bread', 'Milk'], 
                ['Bread', 'Diapers', 'Beer', 'Eggs'], 
                ['Milk', 'Diapers', 'Beer', 'Coke'], 
                ['Bread', 'Milk', 'Diapers', 'Beer'], 
                ['Bread', 'Milk', 'Diapers', 'Coke']]

    def test_candidate1_itemsets(self):
        """ Verify candidate 1-itemsets """
        C1 = create_candidates(self.dataset)

        assert len(C1) == 6

        # check contents
        assert C1[0] == frozenset(['Beer'])
        assert C1[5] == frozenset(['Milk'])

    def test_support_prune(self):
        # Prune candidate 1-itemsets via support-based
        # pruning to generate frequent 1-itemsets.
        C1 = create_candidates(self.dataset)
        F1, support_data = support_prune(self.D, C1, 0.6)

        # Check F1 contents
        assert len(F1) == 4
        assert F1[3] == frozenset(['Bread'])
        assert F1[2] == frozenset(['Milk'])
        assert F1[1] == frozenset(['Beer'])
        assert F1[0] == frozenset(['Diapers'])

        # Check support_data contents
        assert len(support_data) == 6
        assert support_data[frozenset(['Beer'])] == 0.6
        assert support_data[frozenset(['Milk'])] == 0.8

    def test_generate_all_frequent_itemsets(self):
        # Generate all the frequent itemsets using the Apriori algorithm.
        F, support_data = apriori(self.dataset, min_support=0.6)

        # Check F contents
        assert len(F) == 3
        assert len(F[0]) == 4
        assert len(F[1]) == 4
        assert len(F[2]) == 0

        # check support data
        assert support_data[frozenset(['Milk'])] == 0.8
        assert support_data[frozenset(['Bread'])] == 0.8
        assert support_data[frozenset(['Beer'])] == 0.6
        assert support_data[frozenset(['Diapers'])] == 0.8
        assert support_data[frozenset(['Beer', 'Diapers'])] == 0.6
        assert support_data[frozenset(['Diapers','Beer'])] == 0.6
        assert support_data[frozenset(['Milk','Diapers'])] == 0.6
        assert support_data[frozenset(['Milk','Bread'])] == 0.6

    def test_apriori_association_rules(self):
        # Generate the association rules from a list of frequent itemsets.
        F, support_data = apriori(self.dataset, min_support=0.6)
        H = generate_rules(F, support_data, min_confidence=0.8)

        # Check H contents
        assert len(H) == 1
        assert H[0] == (frozenset({'Beer'}),
                        frozenset({'Diapers'}), 1.0)

        assert support_data[frozenset(['Beer','Diapers'])] == 0.6


