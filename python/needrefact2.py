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

        def print_ok():
            print 'ok'
        
        def print_yes():
            print 'yes'
        
        def print_good():
            print 'good'

        def print_faild():
            print 'faild'
        
        def print_opps():
            print 'opps'

        def raise_exception():
            raise Exception()

        args_map = {}
        args_map[('dog', 'eat', 'bone')] = print_ok
        args_map[('dog', 'eat', 'ball')] = print_faild
        args_map[('dog', 'play', 'bone')] = print_ok
        args_map[('dog', 'play', 'ball')] = print_yes
        args_map[('dog', 'play', 'mouse')] = print_opps

        args_map[('dolphin', 'eat', 'bone')] = print_faild
        args_map[('dolphin', 'eat', 'ball')] = print_faild
        args_map[('dolphin', 'play', 'bone')] = print_faild
        args_map[('dolphin', 'play', 'ball')] = print_good

        func = args_map.get((pet, action, target), raise_exception)
        func()

    if __name__ == '__main__':
        import doctest
        doctest.testmod()
