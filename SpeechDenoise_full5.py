import numpy as np
from numpy import linalg as LA
import sys
import librosa
from scipy import linalg
import copy
import random
from math import log
# import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import logging
import argparse



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
        logging.info(
            'Error: Overlapping windows cannot be larger than frame length!')
        sys.exit()
#
    len_signal = len(signal)
    #
    logging.info('The signal length is %s: ' % (len_signal))
    #
    K = np.ceil(len_signal / L).astype('int')  # num_frames
    #
    logging.info('The number of frames \'K\' is %s: ' % (K))
    logging.info('The length of each frame \'L\' is %s: ' % (L))
    #
    X_tmp = []
    k = 1
    while (True):
        start_ind = ((k - 1) * (L - M) + 1) - 1
        end_ind = ((k * L) - (k - 1) * M)
        if start_ind == len_signal:
            break
        elif (end_ind > len_signal):
            # logging.info(('k=%s, [%s, %s] ' % (k, start_ind, end_ind - 1))
            val_in = len_signal - start_ind
            tmp_seg = np.zeros(L)
            tmp_seg[:val_in] = signal[start_ind:]
            X_tmp.append(tmp_seg)
            break
        else:
            # logging.info(('k=%s, [%s, %s] ' % (k, start_ind, end_ind - 1))
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
            self, X, params, M, signal=[]
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
        # THE following K and L were typo/switched in the GAD.py code. they're fixed here:
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

        # The following are (sortof) optional.
        # we use the following 3 instance variables to calculate RMSE after each iter
        self.M = M
        self.signal = signal  # we leave this empty unless we actually want to do the RMSE, which is computationall
        #intense and also requires sending the (big) signal across the line...
        self.rmse = []  # to hold RMSE after each iter
        # and this one to plot solution quality over time
        self.k_min_data = []

    def function(self, S, big_number=25.0):
        # Note: this only works for f(S); it will NOT work on any input except S. to do that, would need to find
        # the elements in the function's argument that are not in S, then iteratively add to the value we return.
        return (len(S) * big_number - self.k_min_sum)

    def functionMarg_quickestimate(self,
                                   new_elements,
                                   curr_elements,
                                   big_number=25.0):
        new_elems = [ele for ele in new_elements if ele not in curr_elements]
        if not len(new_elems):
            return (0)

        # This is a bit of a hack...
        # Actually, the below version is unfair/inaccurate, as we should really update R by orthogonalizing
        # after we add each column. Here, I'm treating the function as a modular function within each
        # round of adaptive sampling (and submodular across rounds) which is entertainingly ridiculous.
        # BUT if it works at de-noising, then I don't care :)
        # NOTE that in the original GAD code, self.k_min_sum is similar to what I'm calling sum_of_norm_ratios.
        new_elems = [ele for ele in new_elements if ele not in curr_elements]
        R_copy = copy.copy(self.R)
        #sum_of_norm_ratios = np.sum(LA.norm(R_copy[:, new_elems], 1)) / np.sum([LA.norm(R_copy[:, I_ind_k_min], 2) for I_ind_k_min in new_elems])
        sum_of_norm_ratios = np.sum([
            LA.norm(R_copy[:, I_ind_k_min], 1) / LA.norm(
                R_copy[:, I_ind_k_min], 2) for I_ind_k_min in new_elems
        ])

        return (len(new_elems) * big_number - sum_of_norm_ratios)

    def functionMarg(self, new_elements, curr_elements, big_number=25.0):
        # This is the more correct (but slower and more complicated) functionMarg. See note in other simpler version above.
        # NOTE: IT ASSUMES THAT S IS THE CURRENT S. IT WILL BE WRONG WHEN input S is NOT the current solution!!!
        new_elems = [ele for ele in new_elements if ele not in curr_elements]
        if not len(new_elems):
            return (0)

        # Copy everything important... we have to update them iteratively for each ele in the new sample, but
        # we might not use this sample so can't change the originals...
        R_copy = copy.copy(self.R)
        #print self.R.shape, '=self.R.shape'
        D_copy = copy.copy(self.D)
        I_copy = copy.copy(self.I)
        k_min_sum_copy = copy.copy(self.k_min_sum)
        set_ind_copy = self.set_ind
        marginal_k_min_sum_copy = 0

        # New elements we compute marginal value for
        new_elems = [ele for ele in new_elements if ele not in curr_elements]

        # do the GAD find_column() routine, but we're not trying to find a new column; we're evaluating
        # the quality of ones in the set we sampled. Basically, that means checking for changes in k_min_sum_copy.
        #
        for I_ind_k_min in new_elems:
            sample_avg_k_min = 0
            r_k = R_copy[:, I_ind_k_min]
            #
            k_min = LA.norm(r_k, 1) / LA.norm(r_k, 2)
            #logging.info('k_min inside a sample is %s: ' % k_min)
            sample_avg_k_min += k_min
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
        #print 'sample avg k_min = ', sample_avg_k_min/np.float(len(new_elems))
        #logging.info('marginal_k_min_sum_copy of a sample is %s: ' % marginal_k_min_sum_copy)
        #logging.info('some sample val is %s: ' % ( - marginal_k_min_sum_copy))
        #logging.info('big number is %s: ' % (  big_number))
        #logging.info('len(new_elems) %s: ' % (  len(new_elems)))
        return (len(new_elems) * big_number - marginal_k_min_sum_copy)


def adaptiveSampling_adam(f,
                          k,
                          numSamples,
                          r,
                          opt,
                          alpha1,
                          alpha2,
                          compute_rmse=False,
                          speed_over_accuracy=False,
                          parallel=False):

    # This large uncommented script is not complicated enough, so here we go:
    if speed_over_accuracy:

        def functionMarg_closure(new_elements, curr_elements, big_number=25.0):
            return f.functionMarg_quickestimate(
                new_elements, curr_elements, big_number=25.0)
    else:

        def functionMarg_closure(new_elements, curr_elements, big_number=25.0):
            return f.functionMarg(new_elements, curr_elements, big_number=25.0)

    S = copy.deepcopy(f.meaningfulNodes)
    X = []

    while len(X) < k and len(S + X) > k:
        currentVal = f.function(X)

        logging.info([
            currentVal, 'ground set remaining:',
            len(S), 'size of current solution:',
            len(X)
        ])
        samples = []
        samplesVal = []

        # PARALLELIZE THIS LOOP it is emb. parallel

        def sample_elements():
            #logging.info(len(S), 'is len(S)'
            sample = sampleS(S, k / r)
            #logging.info(len(S), 'is len(S);', k/r, 'is k/r', k,'is k', r, 'is r', len(sample), 'is len sample'
            sampleVal = functionMarg_closure(sample, X)
            samplesVal.append(sampleVal)
            samples.append(sample)

        if parallel:
            Parallel(n_jobs=num_cores)(
                delayed(sample_elements) for i in range(numSamples))

        else:
            for i in range(numSamples):
                sample_elements()

        maxSampleVal = max(samplesVal)
        #print 'max sample val / len', maxSampleVal/np.float(k/r), 'avg sample val', np.mean(samplesVal)/np.float(k/r)
        #print 'max sample val / len', maxSampleVal, 'avg sample val', np.mean(samplesVal)

        bestSample = samples[samplesVal.index(maxSampleVal)]

        if maxSampleVal >= (opt - currentVal) / (alpha1 * float(r)):
            X += bestSample
            #logging.info(len(X), 'is len(X)'
            for element in bestSample:
                S.remove(element)
                #logging.info(len(S), 'is len(S) after removing an element from best sample'

            # Now we need to do some bookkeeping just for the audio de-noising objective:
            for I_ind_k_min in bestSample:
                r_k_min = f.R[:, I_ind_k_min]
                #tmp.append(LA.norm(r_k, 1) / LA.norm(r_k, 2))
                #
                #k_min = tmp[ind_k_min]
                k_min = LA.norm(r_k_min, 1) / LA.norm(r_k_min, 2)
                #print 'k_min added to soln', k_min
                #                print 'k_min in best', k_min
                f.k_min_data.append(k_min)  # This is just for logging purposes
                #
                f.k_min_sum = f.k_min_sum + k_min
                #logging.info(k_min
                #
                #r_k_min = f.R[:, I_ind_k_min]
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

            if compute_rmse:  # Note the variables below are all temp versions of the 'real' ones.
                D = np.vstack(f.D).T
                I = f.I
                X_t = np.dot(np.dot(D, D.T), f.X)
                s_rec = unbuffer(X_t, f.L - f.M)
                f.rmse.append(
                    linalg.norm(signal - s_rec[0:len(signal)] / np.max(s_rec))
                )  # omitted padding at end of s_rec

            #print 'residual self.R is', f.R.shape, 'ground set I (minframe) is', f.I.shape, 'dictionary D is', np.vstack(f.D).T.shape

        else:
            logging.info(
                "NEED TO DO FILTERING STEP, BUT I HAVEN'T CODED THE functionMARG to handle this yet so breaking"
            )
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
        logging.info('NOT ENOUGH ELEMENTS left in ground set S')
        X = S + X

    #logging.info(f.function(X), len(S), len(X))
    return X


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fraction_to_drop',
        default=0.11,
        type=float,
        help='fraction_to_drop')

    parser.add_argument('--r', default=10, type=int, help='r')

    parser.add_argument('--k', default=80, type=int, help='')

    parser.add_argument('--audio', default='alexa', type=str, help='')

    parser.add_argument('--num_samples', default=36*4, type=int, help='r')

    parser.add_argument('--speed_over_accuracy', default=1, type=int, help='')

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        level='INFO',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        filename='adaptive_%d_%d.log' % (args.num_samples, args.speed_over_accuracy), filemode='w')

    logging.info(args)
    ### Params ###
    audio = args.audio
    L = 512  # frame length
    M = 500  # overlapping windows
    my_n_iter = 100
    parallel = True  # parallelize the inner for loop of adaptive sampling
    # FOR SPEED TESTS do NOTTT compute the rmse. it's super slow. we just use it to plot stuff if we want.
    compute_rmse = False
    fraction_to_drop = args.fraction_to_drop
    k = args.k  # iterations in original GAD algo
    #numSamples = 24
    numSamples = args.num_samples
    r = args.r  # rounds of adaptive sampling
    #r = k+1-1
    opt = 1.0  # small so we don't do filtering subroutine as I haven't written that part :)
    alpha1 = 1.0
    alpha2 = 1.0
    speed_over_accuracy = args.speed_over_accuracy  # IF true, we assume the fn is modular within rounds and speed things up a lot!
    # We do that by using functionMarg instead of functionMarg_better. May sacrifice some performance, though.
    #############

    num_cores = multiprocessing.cpu_count()
    logging.info('num_cores:')
    logging.info(num_cores)

    params = {
        'rule_1': {
            'n_iter': my_n_iter
        },
        'rule_2': {
            'error': 10**-7
        },
        'verbose': True
    }
    if parallel:
        logging.info("Parallelize sampling.")
    else:
        logging.info("Do not parallelize sampling.")

    #
    #signal, fs = librosa.core.load('./dataset/source2.wav', 44100)
    #signal2, fs2 = sf.read('./dataset/source1.wav', samplerate=fs)
    # signal, fs = librosa.core.load('./dataset/alexa_demo.m4a', 44100)
    signal, fs = librosa.core.load('./dataset/' + audio + '.wav', 44100)
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
    n_to_drop = int(fraction_to_drop * signal.shape[0])
    drop_idx = np.random.choice(
        range(signal.shape[0]), n_to_drop, replace=True)
    signal[drop_idx] = 0

    # plt.close('all')
    # plt.figure()
    # plt.plot(signal, 'k', alpha=0.3)
    # plt.plot(signal_original, 'r:', alpha=0.3, linewidth=1.0)
    # plt.legend(('Noisy', 'Clean'))
    # plt.title('')
    # plt.show()

    X_tmp = buffer(signal, L, M)
    X = np.vstack(X_tmp).T.astype('float')

    # Initialize class with the buffered song X and the objective function

    if compute_rmse:
        # FOR PLOT GENERATION USE:
        f = SpeechDenoise(X, params, M, signal)
    else:
        f = SpeechDenoise(X, params, M)
    logging.info("START")
    solution_elements = adaptiveSampling_adam(f, k, numSamples, r, opt, alpha1,
                                              alpha2, compute_rmse,
                                              speed_over_accuracy)

    # Put the output back into the form of the original song
    D_stack = np.vstack(f.D).T
    X_t = np.dot(np.dot(D_stack, D_stack.T), X)
    s_rec = unbuffer(X_t, L - M)

    #print f.rmse
    logging.info("STOP")

    #######################################
    # THIS IS WHERE THE TIMER SHOULD STOP #
    #######################################

    # # PLOTS
    # if compute_rmse:
    #     plt.close('all')
    #     plt.figure()
    #     plt.plot(f.rmse, 'r:', alpha=0.8)
    #     plt.title('RMSE: Original track (without noise) vs. Denoised track')
    #     plt.show()

    # avg_sparsity_of_samples_added_per_round = []
    # for rd in range(r):
    #     idx_left = rd * r
    #     idx_right = rd * r + r
    #     avg_sparsity_of_samples_added_per_round.append(
    #         np.mean(f.k_min_data[idx_left:idx_right]))
    # plt.close('all')
    # plt.figure()
    # plt.plot(avg_sparsity_of_samples_added_per_round, 'b', alpha=0.8)
    # plt.title('avg. sparsity values of elements per round')
    # plt.show()

    # plt.close('all')
    # plt.figure()
    # plt.plot(signal, 'k', alpha=0.3)
    # plt.plot(signal_original, 'r:', alpha=0.3, linewidth=1.0)
    # plt.plot(s_rec / max(s_rec), 'b', alpha=0.3, linewidth=1.0)
    # plt.legend(('Noisy', 'Clean', 'Denoised Estimate'))
    # plt.title('')
    # plt.show()

    # plt.close('all')
    # plt.figure()
    # plt.plot(D_stack[0], 'm', alpha=0.3)
    # plt.title('First dictionary atom (element) added to the solution')
    # plt.show()

    #logging.info('s_rec', s_rec)

    # Output the WAV files. Note we also re-make the original, as encoding degrades (so it's only fair)
    librosa.output.write_wav("original.wav", signal_original, fs)
    librosa.output.write_wav("dataset/noisy_%s.wav" % str(fraction_to_drop), signal,
                             fs)
    librosa.output.write_wav(
        "dataset/adaptive_%s_%s_%d_%d_%d_%d.wav" % (audio, str(fraction_to_drop), k,
                                              r, numSamples, speed_over_accuracy), s_rec / np.max(s_rec), fs)
