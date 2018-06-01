""" Validate behavior of FP Growth. """
import unittest

from gunter.fp_growth import fpgrowth, generate_rules


class TestFpGrowth(unittest.TestCase):
    """ Validate FP Growth algorithms. """

    def setUp(self):
        self.dataset = self.load_dataset()
        self.D = map(set, self.dataset)

    def load_dataset(self):
        """Loads an example of market basket transactions for testing purposes.

        Returns
        -------
        A list (database) of lists (transactions). Each element of a 
        transaction is an item.
        """
        return [['Bread', 'Milk'], 
                ['Bread', 'Diapers', 'Beer', 'Eggs'], 
                ['Milk', 'Diapers', 'Beer', 'Coke'], 
                ['Bread', 'Milk', 'Diapers', 'Beer'], 
                ['Bread', 'Milk', 'Diapers', 'Coke']]


    def test_generate_all_frequent_itemsets(self):
        F, support_data = fpgrowth(self.dataset, min_support=0.6)

        # verify size
        assert len(F) == 2
        assert len(support_data) == 8

        # verify contents
        assert support_data[frozenset(['Bread'])] == 0.8
        assert support_data[frozenset(['Beer'])] == 0.6
        assert support_data[frozenset(['Diapers','Beer'])] == 0.6
        assert support_data[frozenset(['Diapers'])] == 0.8
        assert support_data[frozenset(['Bread','Diapers'])] == 0.6
        assert support_data[frozenset(['Milk', 'Diapers'])] == 0.6
        assert support_data[frozenset(['Milk'])] == 0.8
        assert support_data[frozenset(['Bread', 'Milk'])] == 0.6

    def test_rule_generation(self):
        F, support_data = fpgrowth(self.dataset, min_support=0.6)
        H = generate_rules(F, support_data, min_confidence=0.8)

        assert H[0][0] == frozenset(['Beer'])
        assert H[0][1] == frozenset(['Diapers'])
        assert H[0][2] == 1.0
