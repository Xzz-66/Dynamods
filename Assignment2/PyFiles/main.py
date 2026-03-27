import numpy as np

class Viterbi():
    def __init__(self, init_prob, trans_prob, emis_prob):
        
        self.init_prob = init_prob      # Initial probabilities [E, I]
        self.trans_prob = trans_prob    # Transition probabilities [E, I] [E, I].T
        self.emis_prob = emis_prob      # Emmision probabilities [a u g c] [E I].T
        
    def run_viterbi(self, observation):
        '''
        Help
        '''
        
        def init_delta(observation): 
            '''
            Initialize the recursion by computing delta for t=1 following the euqaiton cap_pi(s)*B(O_{0}|s).
            '''
            return [self.init_prob[s] * self.emis_prob[s, observation] for s in range(2)]
    
        def recursion(previous_delta, observation):
            '''
            Help
            '''
            # Initialise lists to store psi and delta
            psi = []
            delta = []
            
            # Repeat for amount of states (E, I)
            for s in range(2):
                
                # Compute delta_{t-1}(r) * A(r|s) * B(O_{t}|s) for r = (0, 1)
                state_prob = [previous_delta[r] * self.trans_prob[r, s] * self.emis_prob[s, observation] for r in range(2)]
                
                arg_max = np.argmax(state_prob) # Get index of max
                max = state_prob[arg_max]       # Get probability of max
                
                psi.append(arg_max)
                delta.append(max)
                
            return delta, psi 

        def termination(delta, psi):
            '''
            Help
            '''

            path_prob = np.max(delta)
            
            # Get the state of the last 
            state = psi[-1, np.argmax(delta)]
            
            final_state_path = [state]
            for i in range(psi.shape[0]-2, 0, -1):
                state = psi[i, state]
                final_state_path.append(state.copy())
            
            return final_state_path, path_prob
        
        # Initialization
        delta = init_delta(observation=observation[0])
        
        # Recursion
        backtrack = np.zeros((len(observation), 2), dtype=int)
        for t, o in enumerate(observation[1:]):
            delta, psi = recursion(delta, o)
            backtrack[t,:] = psi
    
        # Termination
        final_state_path, path_prob = termination(delta, backtrack)
        
        return final_state_path, path_prob
    
    
    
if __name__ == '__main__':
      
    # Probability matices
    # Emmision probabilities [a u g c] [E I].T
    emission = np.array([[0.25, 0.25, 0.25, 0.25],
                        [0.4, 0.4, 0.05, 0.15]])

    # Transition probabilities [E I] [E, I].T
    transition = np.array([[0.9, 0.1],
                        [0.2, 0.8]])

    # Initial state probablities [E I]
    initial = [0.5, 0.5]

    # 'agcgc'
    patient_alpha = [0,2,3,2,3]
    # 'auuau'
    patient_beta = [0,1,1,0,1]

    model = Viterbi(initial, transition, emission)

    print(model.run_viterbi(patient_alpha))
    print(model.run_viterbi(patient_beta))