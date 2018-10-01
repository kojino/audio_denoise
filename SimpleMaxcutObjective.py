
import numpy as np

def greedy(f, k):
    """ Basic greedy algorithm: for k steps, Find the element among all remaining elements with 
    the greatest marginal value added to the current solution and add this to the current solution.
    INPUTS are the class, f, containing the methods 'function' and its derivative 'functionMarg' 
    that we want to optimize and int k -- the cardinality constraint.
    OUTPUTS are (1) L, a list containing the the solution set ordered in the order we added them 
    and (2) a list F_of_L containing the respective values, such that for example F_of_L[2] is 
    the value of the first 3 items in L."""

    solution_set_S = []
    F_of_S = [0]
    N = [ele for ele in f.meaningfulNodes]
    print('Greedy algorithm will select', k, 'elements from the ground set of', len(N), 'elements')

    for i in range(k):
       
        # Compute the marginal addition for each node in N, 
        elementVals = [ f.functionMarg( solution_set_S, [element] ) for element in N ]
#        print elementVals
        bestVal_idx = np.argmax(elementVals)
#        print bestVal_idx

        # Add the best one to solution; remove it from remaining elements N
        solution_set_S.append( N[bestVal_idx] )
#        print solution_set_S
        N.remove( N[bestVal_idx] )

        # Record the current value of our objective function (for plotting, inspection, etc
        # note this is UNNECESSARY for the algorithm -- just for us to get more info.
        F_of_S.append( f.function(solution_set_S) )
#
    print('Greedy algorithm finished, achieving value', F_of_S[-1])
    return solution_set_S, F_of_S





class SimpleMaxcutObjective:  

    def __init__(self):
        """
        Builds a very simple undirected/Boolean network (adjacency matrix),
        then implements the basic maxcut objective function and it's marginal value function
        """

        # Element names:
        self.meaningfulNodes = list(range(10))

        # Network
        # self.Adj = np.array([[0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        #                      [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        #                      [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        #                      [0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        #                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        #                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        #                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        #                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        #                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.Adj =   np.array([[0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
                               [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                               [1, 1, 0, 1, 0, 0, 1, 0, 0, 0],
                               [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                               [1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
                               [1, 1, 0, 0, 1, 0, 1, 1, 1, 1],
                               [0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                               [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
                               [0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
                               [1, 1, 0, 1, 0, 1, 1, 1, 0, 0]])
          
    # Objective function
    def function(self, solution_set_S):
        # Basic max cut i.e. count the edges with one endpoint in S and one endpoint not in S
        # the 0.5 prevents double counting edges
        return ( 0.5*np.sum(self.Adj[solution_set_S]) + 0.5*np.sum(self.Adj[:,solution_set_S]) - np.sum(self.Adj[solution_set_S][:,solution_set_S]) ) 

    
    # Marginal value function
    def functionMarg(self, S, new_elements):
        return self.function( S+new_elements ) - self.function(S)




if __name__ == '__main__':
    # Generate our class containing the function
    f = SimpleMaxcutObjective()

    # Try it out
    print('value of node 0:', f.function([0])) # the first node has 6 friends, so we should get 6
    print('value of node 0:', f.function([2])) # the 3rd node has 4 friends, so we should get 4

    # But if we've already added node 0, then adding node 2 only gives us 2 more, as we lose the 
    # edge from 0->2  that we already counted and we also cannot add the symmetric edge from 2-> 0 
    # as both of these now in our solution (maxcut has diminishing returns!)
    print('marginal value of node 2 after adding node 0:', f.functionMarg ([0], [2])) 

    # Optimize with greedy
    k = 5 # size of the set we want to find
    solution_set_S, F_of_S = greedy(f, k)
    print(solution_set_S)
    print(F_of_S)







