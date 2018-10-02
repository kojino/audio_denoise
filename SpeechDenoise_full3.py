import numpy as np
from numpy import linalg as LA
import sys
import librosa
from scipy import linalg
import copy
import random
from math import log
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

# print num cores
num_cores = multiprocessing.cpu_count()
print('num_cores:')
print(num_cores)


def sampleS(S, k):
    sample = []
    if len(S) <= k:
        return S
    while len(sample) < k:
        new = S[random.randint(0, len(S) - 1)]
        if not new in sample:
            sample.append(new)
    return sample


def buffer(signal, L, M):
    if M >= L:
        print('Error: Overlapping windows cannot be larger than frame length!')
        sys.exit()
#
    len_signal = len(signal)
    #
    print('The signal length is %s: ' % (len_signal))
    #
    K = np.ceil(len_signal / L).astype('int')  # num_frames
    #
    print('The number of frames \'K\' is %s: ' % (K))
    print('The length of each frame \'L\' is %s: ' % (L))
    #
    X_tmp = []
    k = 1
    while (True):
        start_ind = ((k - 1) * (L - M) + 1) - 1
        end_ind = ((k * L) - (k - 1) * M)
        if start_ind == len_signal:
            break
        elif (end_ind > len_signal):
            # print ('k=%s, [%s, %s] ' % (k, start_ind, end_ind - 1))
            val_in = len_signal - start_ind
            tmp_seg = np.zeros(L)
            tmp_seg[:val_in] = signal[start_ind:]
            X_tmp.append(tmp_seg)
            break
        else:
            # print ('k=%s, [%s, %s] ' % (k, start_ind, end_ind - 1))
            X_tmp.append(signal[start_ind:end_ind])
        k += 1


#
    return X_tmp


def unbuffer(X, hop):
    N, L = X.shape
    #
    T = N + L * hop
    K = np.arange(0, N)
    x = np.zeros(T)
    H = np.hanning(N)
    for k in xrange(0, L):
        x[K] = x[K] + np.multiply(H, X[:, k])
        K = K + hop


#
    return x


class SpeechDenoise:
    def __init__(
            self, X, params
    ):  # X is the np.vstacked transpose of the buffered signal (buffered==split up into overlapping windows)

        self.meaningfulNodes = range(
            X.shape[1])  # this is pretty much the same thing as self.I

        self.X = X
        self.D = []
        self.params = params
        self.n_iter = self.params['rule_1']['n_iter']  # num_iterations
        self.error = self.params['rule_2']['error']  # num_iterations
        #
        self.verbose = self.params['verbose']
        #
        self.K = self.X.shape[0]  # sample length
        self.L = self.X.shape[
            1]  # maximum atoms to be learned (i.e. size of ground set)
        #
        self.I = np.arange(
            0, self.L
        )  # self.I is the ground set of elements (dictionary atoms) we can choose
        self.set_ind = []

        self.k_min_sum = 0.0

        # Initializating the residual matrix 'R' by using 'X'
        self.R = self.X.copy()

    def function(self, S, big_number=100.0):
        # Note: this only works for f(S); it will NOT work on any input except S. to do that, would need to find
        # the elements in the function's argument that are not in S, then iteratively add to the value we return.
        return (self.L * big_number - self.k_min_sum)

    def functionMarg(self, S, T, big_number=100.0):
        if not len(S):
            return (0)
        if not len(T):
            return self.function(S)
        # This is a bit of a hack...
        # Actually, the below version is unfair/inaccurate, as we should really update R by orthogonalizing
        # after we add each column. Here, I'm treating the function as a modular function within each
        # round of adaptive sampling (and submodular across rounds) which is entertainingly ridiculous.
        # BUT if it works at de-noising, then I don't care :)
        # NOTE that in the original GAD code, self.k_min_sum is similar to what I'm calling sum_of_norm_ratios.
        new_elems = [ele for ele in T if ele not in S]
        sum_of_norm_ratios = np.sum(LA.norm(new_elems, 1)) / np.sum(
            [LA.norm(r_k, 2) for r_k in new_ele])
        return (len(new_ele) * big_number - sum_of_norm_ratios)

    def functionMarg_better(self, S, T, big_number=100.0):
        # This is the more correct (but slower and more complicated) functionMarg. See note in other simpler version above.
        # NOTE: IT ASSUMES THAT S IS THE CURRENT S. IT WILL BE WRONG WHEN input S is NOT the current solution!!!
        if not len(S):
            return (0)
        if not len(T):
            return self.function(S)

        # Copy everything important... we have to update them iteratively for each ele in the new sample, but
        # we might not use this sample so can't change the originals...
        # R_copy         = self.R.copy()
        # D_copy         = self.D.copy()
        # I_copy         = self.I.copy()
        # k_min_sum_copy = self.k_min_sum.copy()
        # set_ind_copy   = self.set_ind
        # marginal_k_min_sum_copy = 0
        R_copy = copy.copy(self.R)
        D_copy = copy.copy(self.D)
        I_copy = copy.copy(self.I)
        k_min_sum_copy = copy.copy(self.k_min_sum)
        set_ind_copy = self.set_ind
        marginal_k_min_sum_copy = 0

        # New elements we compute marginal value for
        new_elems = [ele for ele in T if ele not in S]

        # do the GAD find_column() routine, but we're not trying to find a new column; we're evaluating
        # the quality of ones in the set we sampled. Basically, that means checking for changes in k_min_sum_copy.
        #tmp = []
        #
        for I_ind_k_min in new_elems:
            r_k = R_copy[:, I_ind_k_min]

            #
            k_min = LA.norm(r_k, 1) / LA.norm(r_k, 2)
            #
            marginal_k_min_sum_copy = marginal_k_min_sum_copy + k_min
            k_min_sum_copy = k_min_sum_copy + k_min
            #
            r_k_min = R_copy[:, I_ind_k_min]
            #
            # Set the l-th atom to equal to normalized r_k
            psi = r_k_min / LA.norm(r_k_min, 2)
            #
            # Add to the dictionary D and its index and shrinking set I
            D_copy.append(psi)
            set_ind_copy.append(I_ind_k_min)

            # Compute the new residual for all columns k
            for kk in I_copy:
                r_kk = R_copy[:, kk]
                alpha = np.dot(r_kk, psi)
                R_copy[:, kk] = r_kk - np.dot(psi, alpha)

    #
            I_copy = np.delete(I_copy, [I_ind_k_min])

        return (len(new_elems) * big_number - marginal_k_min_sum_copy)


def adaptiveSampling_adam(f,
                          k,
                          numSamples,
                          r,
                          opt,
                          alpha1,
                          alpha2,
                          parallel=False):

    S = copy.deepcopy(f.meaningfulNodes)
    X = []

    while len(X) < k and len(S + X) > k:
        currentVal = f.function(X)

        print currentVal, 'ground set remaining:', len(
            S), 'size of current solution:', len(X)
        samples = []
        samplesVal = []

        # PARALLELIZE THIS LOOP it is emb. parallel
        def sample_elements():
            #print len(S), 'is len(S)'
            sample = sampleS(S, k / r)
            #print len(S), 'is len(S);', k/r, 'is k/r', k,'is k', r, 'is r', len(sample), 'is len sample'

            #sampleVal = f.functionMarg(sample, X)
            sampleVal = f.functionMarg_better(sample, X)
            samplesVal.append(sampleVal)
            samples.append(sample)

        if parallel:
            Parallel(n_jobs=num_cores)(delayed(sample_elements) for i in range(numSamples))

        else:
            for i in range(numSamples):
                sample_elements()

        
        maxSampleVal = max(samplesVal)
        bestSample = samples[samplesVal.index(maxSampleVal)]

        if maxSampleVal >= (opt - currentVal) / (alpha1 * float(r)):
            X += bestSample
            #print len(X), 'is len(X)'
            for node in bestSample:
                S.remove(node)
                #print len(S), 'is len(S) after removing an element from best sample'

            # Now we need to do some bookkeeping just for the audio de-noising objective:
            for I_ind_k_min in bestSample:
                r_k = f.R[:, I_ind_k_min]
                #tmp.append(LA.norm(r_k, 1) / LA.norm(r_k, 2))
                #
                #k_min = tmp[ind_k_min]
                k_min = LA.norm(r_k, 1) / LA.norm(r_k, 2)
                #
                f.k_min_sum = f.k_min_sum + k_min
                #print k_min
                #
                r_k_min = f.R[:, I_ind_k_min]
                #
                # Set the l-th atom to equal to normalized r_k
                psi = r_k_min / LA.norm(r_k_min, 2)
                #
                # Add to the dictionary D and its index and shrinking set I
                f.D.append(psi)
                f.set_ind.append(I_ind_k_min)
                #
                # Compute the new residual for all columns k
                for kk in f.I:
                    r_kk = f.R[:, kk]
                    alpha = np.dot(r_kk, psi)
                    f.R[:, kk] = r_kk - np.dot(psi, alpha)
    #
                f.I = np.delete(f.I, [I_ind_k_min])

        else:
            print "NEED TO DO FILTERING STEP, BUT I HAVEN'T CODED THE functionMARG to handle this yet so breaking"
            break

            # newS = copy.deepcopy(S)
            # samples = []

            # for i in xrange(numSamples/200):
            #     samples.append(sampleS(S,k/r))

            # for element in S:
            #     sumMargElements = 0
            #     count = 0
            #     for sample in samples:
            #         if not element in sample:
            #             sumMargElements += f.functionMarg([element], sample + X)
            #             count += 1

            #     if sumMargElements / count < (opt - currentVal) / (alpha2*float(k)):
            #         newS.remove(element)

            # S = newS

    if len(S + X) <= k:
        print('NOT ENOUGH ELEMENTS left in ground set S')
        X = S + X
    #currentVal = evalByCompleting(f,S,X,k, numSamples)
    print f.function(X), len(S), len(X)

    return X


if __name__ == '__main__':
    L = 512  # frame length
    M = 500  # overlapping windows
    my_n_iter = 100
    parallel = True  # parallelize the inner for loop of adaptive sampling
    if parallel:
        print("Parallelize sampling.")
    else:
        print("Do not parallelize sampling.")
        
    params = {
        #
        'rule_1': {
            'n_iter': my_n_iter  # n_iter
        },
        #
        'rule_2': {
            'error': 10**-7
        },
        #
        'verbose': True
    }
    #
    #signal, fs = librosa.core.load('./dataset/source2.wav', 44100)
    #signal2, fs2 = sf.read('./dataset/source1.wav', samplerate=fs)
    signal, fs = librosa.core.load('./dataset/alexa_demo.m4a', 44100)
    #signal, fs = librosa.core.load('./dataset/pianoscalesong.mp3', 44100)
    #
    signal_original = signal.copy()
    # # Normalize the signal
    # normalizing = linalg.norm(signal)
    # signal /= normalizing
    #
    # # Make some noise
    # # creating noisy mix
    # rng = np.random.RandomState(42)
    # noise = rng.randn(*signal.shape)
    # noise *= 0.3 / linalg.norm(noise)
    # signal = signal+noise
    #
    # Signal drop noise:
    fraction_to_drop = 0.11
    n_to_drop = int(fraction_to_drop * signal.shape[0])
    drop_idx = np.random.choice(
        range(signal.shape[0]), n_to_drop, replace=True)
    signal[drop_idx] = 0

    X_tmp = buffer(signal, L, M)

    # new matrix LxK
    X = np.vstack(X_tmp).T.astype('float')

    # alg = GAD(X, params)
    # D, I = alg.iterative_GAD()

    # Initialize class with the buffered song X and the objective function
    f = SpeechDenoise(X, params)
    k = 100  # iterations in original GAD algo
    numSamples = 20
    print('HELLO')
    r = 10  # rounds of adaptive sampling
    opt = 1.0  # small so we don't do filtering subroutine as I haven't written that part :)
    alpha1 = 1.0
    alpha2 = 1.0
    solution_elements = adaptiveSampling_adam(f, k, numSamples, r, opt, alpha1,
                                              alpha2)

    D_stack = np.vstack(f.D).T
    X_t = np.dot(np.dot(D_stack, D_stack.T), X)
    s_rec = unbuffer(X_t, L - M)

    # plt.figure(1)
    # plt.title('Original signal')
    # plt.plot(signal)

    # plt.figure(2)
    # plt.title('Reconstructed signal')
    # plt.plot(s_rec)

    plt.close('all')
    plt.figure()
    plt.plot(signal, 'k', alpha=0.3)
    plt.plot(signal_original, 'r:', alpha=0.3, linewidth=1.0)
    plt.plot(s_rec, 'b', alpha=0.3, linewidth=1.0)
    plt.legend(('Noisy', 'Clean', 'Denoised Estimate'))
    plt.title('')

    # #print 's_rec', s_rec
    # librosa.output.write_wav("out_librosa_original2.wav", signal_original, fs)
    # librosa.output.write_wav("out_librosa_noisy.wav", signal*normalizing, fs)
    # librosa.output.write_wav("out_librosa_greedy"+ str(my_n_iter) +".wav", s_rec*normalizing, fs)

    librosa.output.write_wav("out_librosa_original2.wav", signal_original, fs)
    librosa.output.write_wav("out_librosa_noisy.wav", signal, fs)
    librosa.output.write_wav("out_librosa_adaptive" + str(my_n_iter) + ".wav",
                             s_rec, fs)
