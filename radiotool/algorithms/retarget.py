# functionality to perform simple retargeting and constrained retargeting

# a song will have a per-beat analysis a priori

def retarget(song, duration, music_labels, out_labels):

    # labels can be array, array of arrays, or function

    # for now, assume music_labels and out_labels are time functions

    target = [out_labels(i) for i in range(int(duration))]
    start = [music_labels(i) for i in range(int(song.duration))]
