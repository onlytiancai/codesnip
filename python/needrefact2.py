    # -*- coding: utf-8 -*-
    '''Please let code becomes much simpler and easier to maintain.
    '''


    def process(pet, action, target):
        '''
        >>> process('dog', 'eat', 'bone')
        ok
        >>> process('dog', 'eat', 'ball')
        faild
        >>> process('dog', 'play', 'ball')
        yes
        >>> process('dog', 'play', 'bone')
        ok
        >>> process('dolphin', 'play', 'ball')
        good
        >>> process('dolphin', 'play', 'bone')
        faild
        >>> process('dolphin', 'eat', 'bone')
        faild
        >>> process('dog', 'play', 'mouse')
        opps
        >>> process('cat', 'catch', 'mouse')
        Traceback (most recent call last):
            ...
        Exception
        '''
        if pet == 'dog':
            if action == 'eat':
                if target == 'bone':
                    print 'ok'
                elif target == 'ball':
                    print 'faild'
                else:
                    raise Exception()
            elif action == 'play':
                if target == 'bone':
                    print 'ok'
                elif target == 'ball':
                    print 'yes'
                else:
                    print 'opps' 
            else:
                raise Exception()
        elif pet == 'dolphin':
            if action == 'eat':
                if target == 'bone':
                    print 'faild'
                elif target == 'ball':
                    print 'faild'
                else:
                    raise Exception()
            elif action == 'play':
                if target == 'bone':
                    print 'faild'
                elif target == 'ball':
                    print 'good'
                else:
                    raise Exception()
            else:
                raise Exception()
        else:
            raise Exception()

    if __name__ == '__main__':
        import doctest
        doctest.testmod()
