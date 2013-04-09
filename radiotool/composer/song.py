class Song(Track):

    def __init__(self, fn, name="Song name"):
        Track.__init__(self, fn, name)
        
    def magnitude_spectrum(self, window):
        """Compute the magnitude spectra"""
        return N.abs(N.fft.rfft(window))
        
    def partial_mfcc(self, window):
        """partial mfcc calculation (stopping before mel band filter)"""
  
        dump_out["names"] = ('MFCC euclidean distance',
                             'RMS energy distance',
                             'Chromagram COSH distance',
                             'Chromagram euclidean distance',
                             'Tempo difference',
                             'Magnitude spectra COSH distance',
                             'RMS energy')
   
    def refine_cut_by(self, refinement, cut_point, window_size=4):
        if refinement == "RMS energy distance":
            return self.refine_cut_rms_jump(cut_point, window_size)
        elif refinement == "MFCC euclidean distance":
            return self.refine_cut_mfcc_euc(cut_point, window_size)
        elif refinement == "Chromagram euclidean distance":
            return self.refine_cut_chroma_euc(cut_point, window_size)
            
        return self.refine_cut_rms_jump(cut_point, window_size)
    
    def refine_cut_rms_jump(self, cut_point, window_size=4):
        # subwindow length
        swlen = 0.250 # 250ms 
        
        start_frame = int((cut_point - window_size * 0.5) * self.sr())
        if (start_frame < 0):
            start_frame = 0
        
        if (start_frame + window_size * self.sr() > self.total_frames()):
            start_frame = self.total_frames() - window_size * self.sr() - 1
            
        self.set_frame(start_frame)
        tmp_frames = self.read_frames(window_size * self.sr())
        
        subwindow_n_frames = swlen * self.sr()
        
        # add left and right channels
        frames = N.empty(window_size * self.sr())
        for i in range(len(frames)):
            frames[i] = tmp_frames[i][0] + tmp_frames[i][1]
        segments = segmentaxis.segment_axis(frames, subwindow_n_frames, axis=0,
        overlap=int(subwindow_n_frames * 0.5))  
        
        RMS_energies = N.apply_along_axis(RMS_energy, 1, segments) 
           
        energy_diffs = N.zeros(len(RMS_energies))
        energy_diffs[1:] = RMS_energies[1:] - RMS_energies[:-1]
        idx = N.where(energy_diffs == max(energy_diffs))[0][0]
        return round(cut_point - window_size * 0.5 +
                           idx * swlen * 0.5, 2), \
               normalize_features(energy_diffs)

    def refine_cut_mfcc_euc(self, cut_point, window_size=4):
        return self.refine_cut_mfcc(cut_point, window_size, "euclidean")
    
    def refine_cut_mfcc(self, cut_point, window_size=4, dist="euclidean"):
        # subwindow length
        swlen = 0.250 #  
        
        self.set_frame(int((cut_point - window_size * 0.5) * self.sr()))
        tmp_frames = self.read_frames(window_size * self.sr())
        
        subwindow_n_frames = swlen * self.sr()
        
        # add left and right channels
        frames = N.empty(window_size * self.sr())
        for i in range(len(frames)):
            frames[i] = tmp_frames[i][0] + tmp_frames[i][1]
        
        segments = segmentaxis.segment_axis(frames, subwindow_n_frames, axis=0,
        overlap=int(subwindow_n_frames * 0.5))
        # compute MFCCs, compare Euclidean distance
        m = mfcc.MFCC(samprate=self.sr(), wlen=swlen)
        mfccs = N.apply_along_axis(m.frame2s2mfc, 1, segments)
        mfcc_dists = N.zeros(len(mfccs))
        for i in range(1,len(mfcc_dists)):
            if dist == "euclidean":
                mfcc_dists[i] = N.linalg.norm(mfccs[i-1] - mfccs[i])
            elif dist == "cosine":
                mfcc_dists[i] = distance.cosine(mfccs[i-1], mfccs[i])
        if DEBUG: print "MFCC euclidean distances: ", mfcc_dists
        idx = N.where(mfcc_dists == max(mfcc_dists))[0][0]
        return round(cut_point - window_size * 0.5 +
                           idx * swlen * 0.5, 2), \
               normalize_features(mfcc_dists)
                           
    def refine_cut_chroma_euc(self, cut_point, window_size=4):
        # subwindow length
        swlen = 0.24 #  
        
        self.set_frame(int((cut_point - window_size * 0.5) * self.sr()))
        tmp_frames = self.read_frames(window_size * self.sr())
        
        subwindow_n_frames = swlen * self.sr()
        
        # add left and right channels
        frames = N.empty(window_size * self.sr())
        for i in range(len(frames)):
            frames[i] = tmp_frames[i][0] + tmp_frames[i][1]
        
        segments = segmentaxis.segment_axis(frames, subwindow_n_frames, axis=0,
        overlap=int(subwindow_n_frames * 0.5))
        # compute chromagram
        fftlength = 44100 * swlen
        # this compute with 3/4 overlapping windows and we want
        # 1/2 overlapping, so we'll take every other column
        cgram = matlab.chromagram_IF(frames, 44100, fftlength)
        # don't need to get rid of 3/4 overlap because we're using it
        # on its own
        # cgram_idx = range(0, len(cgram[0,:]), 2)
        # cgram = cgram[:,cgram_idx]
        cgram_euclidean = N.array([N.linalg.norm(cgram[:,i] - cgram[:,i+1])
                                  for i in range(len(cgram[0,:])-1)])
        idx = N.where(cgram_euclidean == max(cgram_euclidean))[0][0]
        return round(cut_point - window_size * 0.5 +
                     (idx + 1) * swlen * .25, 2), ()
        
    def refine_cut(self, cut_point, window_size=2, scored=True):
        # these should probably all be computed elsewhere and merged
        # (scored?) here
        
        cut_idx = {}
        features = {}
        
        # subwindow length
        swlen = 0.1 # 100ms 
        
        self.set_frame(int((cut_point - window_size * 0.5) * self.sr()))
        tmp_frames = self.read_frames(window_size * self.sr())
        
        subwindow_n_frames = swlen * self.sr()
        
        # add left and right channels
        frames = N.empty(window_size * self.sr())
        for i in range(len(frames)):
            frames[i] = tmp_frames[i][0] + tmp_frames[i][1]
        
        segments = segmentaxis.segment_axis(frames, subwindow_n_frames, axis=0,
                                     overlap=int(subwindow_n_frames * 0.5))

        # should I not use the combined left+right for this feature?
        RMS_energies = N.apply_along_axis(RMS_energy, 1, segments)
        
        if DEBUG: print "RMS energies: ", RMS_energies
        # this is probably not a great feature
        #features["rms_energy"] = RMS_energies
        cut_idx["rms_energy"] = N.where(RMS_energies == max(RMS_energies))[0][0]
        
        ## do it by biggest jump between windows instead
        ## disregard overlapping windows for now
        energy_diffs = N.zeros(len(RMS_energies))
        energy_diffs[1:] = RMS_energies[1:] - RMS_energies[:-1]
        if DEBUG: print "energy differences: ", energy_diffs
        features["rms_jump"] = energy_diffs
        cut_idx["rms_jump"] = N.where(energy_diffs == max(energy_diffs))[0][0]
        
        # compute power spectra, compare differences with I-S distance
        magnitude_spectra = N.apply_along_axis(self.magnitude_spectrum,
                                               1, segments)
        #IS_ms_distances = N.zeros(len(magnitude_spectra))
        # is there a better way... list comprehensions with numpy?
        # for i in range(1,len(IS_ms_distances)):
        #     # not symmetric... do average?
        #     IS_ms_distances[i] = COSH_distance(magnitude_spectra[i-1],
        #                                   magnitude_spectra[i])
        IS_ms_distances = N.array([
            COSH_distance(magnitude_spectra[i],
                          magnitude_spectra[i+1])
            for i in range(len(magnitude_spectra)-1)])
        IS_ms_distances = N.append(IS_ms_distances, 0)
        
        if DEBUG: print "IS ms distances", IS_ms_distances
        features["magnitude_spectra_COSH"] = IS_ms_distances
        cut_idx["magnitude_spectra_COSH"] = N.where(
                IS_ms_distances == max(IS_ms_distances))[0][0] + 1
                
        # compute MFCCs, compare Euclidean distance
        m = mfcc.MFCC(samprate=self.sr(), wlen=swlen)
        mfccs = N.apply_along_axis(m.frame2s2mfc, 1, segments)
        mfcc_dists = N.zeros(len(mfccs))
        for i in range(1,len(mfcc_dists)):
            mfcc_dists[i] = N.linalg.norm(mfccs[i-1] - mfccs[i])
        if DEBUG: print "MFCC euclidean distances: ", mfcc_dists
        features["mfcc_euclidean"] = mfcc_dists
        cut_idx["mfcc_euclidean"] = N.where(mfcc_dists ==
                                            max(mfcc_dists))[0][0]
        
         
        
        combined_features = N.zeros(len(segments))
        for k, v in features.iteritems():
            combined_features += (v - min(v)) / (max(v)- min(v))
        
        cut_idx["combined"] = N.where(combined_features == 
                                      max(combined_features))[0][0]
        if DEBUG: print 'Combined features: ', combined_features
        
        IDX = 'mfcc_euclidean'
        if DEBUG: print "Using ", IDX
                                  
        for k, v in cut_idx.iteritems():
            cut_idx[k] = round(cut_point - window_size * 0.5 +
                               v * swlen * 0.5, 2)
            
        from pprint import pprint            
        if DEBUG: pprint(cut_idx)
        
        # log results to DB for later comparison
        if LOG_TO_DB:
            try:
                con = MySQLdb.connect('localhost', 'root',
                                      'qual-cipe-whak', 'music')
                cur = con.cursor(MySQLdb.cursors.DictCursor)
                desc = "Highest MFCC euclidean distance " + \
                       "with 4 second window, .1 second subwindow and " + \
                       "euclidean distance MFCC segmentation (4 second window)"
                method_q = "SELECT * FROM methods WHERE description = '%s'" \
                            % desc
                cur.execute(method_q)
                method = cur.fetchone()
            
                if method is None:
                    query = "INSERT INTO methods(description) VALUES('%s')" \
                            % desc
                    cur.execute(query)
                    cur.execute(method_q)
                    method = cur.fetchone()
                
                method_id = method["id"]
            
                fn = '.'.join(self.filename.split('/')[-1]
                              .split('.')[:-1]) + '%'
                song_q = "SELECT * FROM songs WHERE filename LIKE %s"
                cur.execute(song_q, fn)
                song = cur.fetchone()
            
                if song is None:
                    print "Could not find song in db matching filename %s" % (
                        filename)
                    return cut_idx[IDX]

                song_id = song["id"]

                result_q = "INSERT INTO results(song_id, song_cutpoint, " + \
                           "method_id) VALUES(%d, %f, %d)" % (song_id, 
                           cut_idx[IDX], method_id)
                cur.execute(result_q)
            
            except MySQLdb.Error, e:
                print "Error %d: %s" % (e.args[0], e.args[1])

            finally:
                if cur:
                    cur.close()
                if con:
                    con.commit()
                    con.close()
        
        return cut_idx[IDX]
    