def checkEqual(f1, f2, num_v):
    def dfs(i, cur_list):
        if i == num_v:
            print(cur_list)
            return f1(*cur_list) == f2(*cur_list)
        ans = True
        for j in range(2):
            ans &= dfs(i+1, cur_list + [j])
        
        return ans
    
    return dfs(0, [])

def generateTable(f, num_v, labels=None):
    if labels != None:
        print(*labels)
    def dfs(i, cur_list):
        if i == num_v:
            print(*cur_list,end=" ")
            values = f(*cur_list)
            for v in values:
                print(v, end=" ")
            print()
            return
        for j in range(2):
            dfs(i+1, cur_list + [j])
    dfs(0, [])
    
def half_adder(x1, x2):
    summ = x1 ^ x2
    carry = x1 and x2 
    return summ, carry
    
def full_adder(x1, x2, c):
    
    summ1, carry1 = half_adder(x1, x2)
    summ2, carry2 = half_adder(summ1, c)
    
    return summ2, (carry1 or carry2)
    
generateTable(full_adder, 3, ["1", "2", "c", "s", "c"])


            
        
    