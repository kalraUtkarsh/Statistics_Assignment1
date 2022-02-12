import math
import numpy as np
from Buffer import Buffer


class Buffer(object):
    def __init__(self, k, sorted):
        self.k = k
        self.clear()
        self.sorted = sorted

    def clear(self):
        self.buffer = np.empty((self.k, 1), dtype=np.int32) 
        self.weight = 0
        self.level = 0 # Associate with each buffer X an integer L(X) that denotes its level.
        self.empty = 0

    def store(self, k_elements):
        '''
        param k_elements: numpy array (kx1)
        '''
        self.buffer = k_elements
        if(self.sorted):
            self.buffer = np.sort(self.buffer)
        # print(f"In buffer: {self.buffer}")

    def copy(self, buffer):
        self.buffer = deepcopy(buffer.buffer)
        self.weight = buffer.weight
        self.level = buffer.level
        self.empty = buffer.empty

    def __getitem__(self,key):
        return self.buffer[key]

    def set_weight(self, w):
        self.weight = int(w)
    def set_level(self, l):
        self.level = l
    def set_buffer_empty(self):
        self.empty = 0
    def set_buffer_full(self):
        self.empty = 1


class Generator:

    def __init__(self, static_range=False):
        self.count = 0 # for generator style generation
        self.range = lambda n: 1000 if static_range else n*1000

    def generate(self, N=1e6):
        N = int(N)
        l = np.random.randint(self.range(N), size=N)   
        return l

    def generate2(self, N=1e6):
        N = int(N)
        if(self.count == N):
            return -1
        yield randint(0,self.range(N))

class buffer_manipulator:
    def __init__(self, fixed_length=2):
        self.length = fixed_length
        self.buffer = []

    def push(self, e):
        self.buffer.append(e)

        if(len(self.buffer) >= self.length):
            self.pop()

        return e

    def pop(self):
        e = self.buffer[0]
        del self.buffer[0]
        return  e

    def get_sum(self):
        return sum(self.buffer)

class Operations():

    def __init__(self, N=1e6, b=10, k=50, buffers_sorted=True, max_range=1e6):
        self.N = int(N)
        self.b = b  # number of buffers
        self.k = k  # number of elements in each buffer
        self.num_collapsed = 0 # number of collapsed operations
        self.sum_weight_out_collapsed = 0 # sum of the weights of the output from the collapse operation
        self.sum_offset_collapsed = 0     # summ of the offsets from the collapse operation
        self.buffers_sorted = buffers_sorted
        self.buffers = [self._create_buffer() for i in range(b)]#np.ndarray((10,),dtype=np.object)
        self.inf = int(max_range) + 1 #np.inf
        self.last_collapse_types = FixedLengthFIFO()
        self.collapse_even_type = 1
        

    def new(self, buffer:Buffer, input_sequence:list):
        seq = input_sequence[:k]
        if(len(input_sequence) < k):
            for i in range(k-len(input_sequence)//2):
                seq.append(-math.inf)
            for i in range(k-len(input_sequence)//2):
                seq.append(math.inf)
            self.leftovers = k-len(input_sequence) 

        buffer.store(seq)
        buffer.set_weight(1)
        buffer.set_buffer_full()


    def collapse(self, buffers:list, level=None):

            c = len(buffers)
            sorted_data = self._merge_buffers(buffers)
            Y_weight = sum([b.weight for b in buffers])
            Y_buffer = self._create_buffer()
            offset = None
            if Y_weight%2 == 0:
                self.last_collapse_types.push(self.collapse_even_type)
                if(self.collapse_even_type == 1):
                    print("Collapse Even 1 type")
                    offset = (output_weight)//2

                # 2. positions = jw(Y) + (w(Y)+2)/2
                elif(self.collapse_even_type == 2):
                    print("Collapse Even 2 type")
                    offset = (output_weight+2)//2

                if(self.last_collapse_types.get_sum() == 2 or self.last_collapse_types.get_sum() == 4):
                    self.collapse_even_type = (self.collapse_even_type%2)+1
            else:
            
    
            
            if(output_weight % 2): # odd
                print("Collapse Odd type")
                offset = (output_weight+1)//2
                self.last_collapse_types.push(4) # As it is binary code (1,2,4) 1-> first choice of even collapse, 2 second choice of even collapse, and 4 odd collapse
            else:
                self.last_collapse_types.push(self.collapse_even_type)
                # 1. positions = jw(Y) + w(Y)/2
                if(self.collapse_even_type == 1):
                    print("Collapse Even 1 type")
                    offset = (output_weight)//2

                # 2. positions = jw(Y) + (w(Y)+2)/2
                elif(self.collapse_even_type == 2):
                    print("Collapse Even 2 type")
                    offset = (output_weight+2)//2

                # Alternate between the two choices when we have successive even collapses with the same choice
                if(self.last_collapse_types.get_sum() == 2 or self.last_collapse_types.get_sum() == 4):
                    self.collapse_even_type = (self.collapse_even_type%2)+1 # change it to 2 if it is 1 and to 1 if it is 2
                # for j=0,1,...k-1
            indcies = [j*output_weight+offset-1 for j in range(self.k)]
            Y_elements = sorted_elements[indcies]
            Y_buffer.store(output_elements.reshape((-1,1)))
            print(f"Collapse -> Output: {list(output_buffer.buffer.reshape((-1)))}")
            self.num_collapsed += 1
            self.sum_weight_out_collapsed += output_weight#np.sum(output_elements)
            self.sum_offset_collapsed += offset
            lemma1_assertion = self.sum_offset_collapsed >= ((self.num_collapsed + self.sum_weight_out_collapsed -1)/2)
            print(f"Lemma 1 ({lemma1_assertion}): {self.sum_offset_collapsed} >= {((self.num_collapsed + self.sum_weight_out_collapsed -1)/2)}")
            assert lemma1_assertion # Lemma 1
            if(level is not None):
                output_buffer.set_level(level)
            
            for buffer in buffers:
                buffer.clear()
            
            buffers[0].copy(output_buffer)
            return output_buffer

    def create_buffer(self):
        buffer = Buffer()
        return buffer

    def _merge_buffers(self, buffers):
        c = len(buffers)
        all_elements = []        
        if(self.buffers_sorted):
            pointers = [0 for _ in range(c)]
            get_element = lambda i: buffers[i][pointers[i]][0]

            for i in range(self.k*c):
                mn_pointer_idx = np.argmin(pointers)
                for i_p in range(c):
                    if(pointers[i_p] >= self.k):
                        continue
                    # printt(f"Min: {mn_pointer_idx}")
                    mn_element = get_element(mn_pointer_idx)
                    # printt(i_p)
                    current_element = get_element(i_p)
                    if(current_element < mn_element):
                        mn_pointer_idx = i_p
                element = get_element(mn_pointer_idx) #self.buffers[mn_pointer_idx][pointers[mn_pointer_idx]]
                all_elements.extend([element]*buffers[mn_pointer_idx].weight)

                pointers[mn_pointer_idx] += 1
                
            all_elements = np.array(all_elements).reshape((-1))
            sorted_elements = np.sort(all_elements)
            # printt(sorted_elements)
            return sorted_elements


    def Collapse(buffers,level = None):
        Y_weight = sum([buffer.weight for buffer in buffers])
        Y_buffer = self.create_buffer()


    def output(self, buffers, phi):
        
        # Similar to COLLAPSE, this operator makes w(Xi) copies of each element in Xi and sorts all the input buffers together, taking the multiple copies of each element into account. 
        sorted_elements = self._merge_buffers(buffers)
        # W = w(X1) + w(X2) + . . . + w(Xc). 
        W = sum([b.weight for b in buffers])
        # The output is the element in position ceil[\phi`kW]
        beta = (self.N+self.leftovers)/self.N # 1+self.leftovers/self.N -> N is too much and leftovers is too small most probably will be numerical errors here (I am not sure) 
        printt(f"Beta={beta}")
        phi_approx = (2*phi+beta-1)/(2*beta) #phi # TODO: Find out what is the relation between phi and phi_approx (phi: real dataset, phi_approx: dataset augmented with -inf and +inf added to the last buffer)
        printt(f"\phi={phi}, \phi`={phi_approx}")
        idx = ceil(phi_approx * self.k * W) - 1 # Zero index and the formula was for 1-indexed
        return int(sorted_elements[idx])



