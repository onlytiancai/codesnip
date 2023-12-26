def trans(src, dst, pair):
    results, remain=[],[]
    print(f'pair:{pair}, src:{src}, dst:{dst}')
    if pair[0] < src[0] and pair[1] > pair[0]:
        print('111', pair)
        remain.append([pair[0], min(pair[1], src[0]-1)])
        pair[0] = min(pair[1], src[0]-1)+1
    if pair[0] >= src[0] and pair[0] <= src[1] and pair[1] > pair[0]:
        begin, end = max(pair[0],src[0]), min(pair[1], src[1])
        print('222', pair, begin, end)
        results.append([dst[0]+begin-src[0], dst[0]+end-src[0]])
        pair[0] = min(pair[1], src[1])+1
    if pair[1] > pair[0]:
        print('333', pair)
        remain.append(pair)
    return results, remain 

if __name__ == '__main__':
    print(trans([3,5],[1,3],[1,10]))
    print(trans([3,5],[1,3],[1,2]))
    print(trans([3,5],[1,3],[5,8]))
    print(trans([3,5],[1,3],[6,8]))
    print(trans([3,5],[1,3],[4,5]))
