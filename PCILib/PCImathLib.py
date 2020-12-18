# -*- coding: UTF-8 -*-
"""
PCI math Lib

请使用pip install sympy

大整数请用python内建类型int

大浮点数请用decimal.Decimal或sympy.Float,比如Decimal('3.1415926535')或Float('1e-3', 3)
"""

import math
import random

try:
    import sympy
    from sympy import oo  # oo是无穷大
    from sympy.abc import x
except ImportError:
    print('missing lib sympy')
    pass


class Cartesian(object):
    """
    Python 计算笛卡尔积

    计算多个集合的笛卡尔积，有规律可循，算法和代码也不难，但是很多语言都没有提供直接计算笛卡尔积的方法，需要自己写大段大段的代码计算笛卡尔积，python 提供了一种最简单的计算笛卡称积的方法(只需要一行代码)，详见下面的代码：
    >>> car = Cartesian([1, 2, 3, 4])
    >>> car.add_data([5, 6, 7, 8],[9, 10, 11, 12])
    >>> print(car.build(return_list=True))
    [(1, 5, 9), (1, 5, 10), (1, 5, 11), (1, 5, 12), (1, 6, 9), (1, 6, 10), (1, 6, 11), (1, 6, 12), (1, 7, 9), (1, 7, 10), (1, 7, 11), (1, 7, 12), (1, 8, 9), (1, 8, 10), (1, 8, 11), (1, 8, 12), (2, 5, 9), (2, 5, 10), (2, 5, 11), (2, 5, 12), (2, 6, 9), (2, 6, 10), (2, 6, 11), (2, 6, 12), (2, 7, 9), (2, 7, 10), (2, 7, 11), (2, 7, 12), (2, 8, 9), (2, 8, 10), (2, 8, 11), (2, 8, 12), (3, 5, 9), (3, 5, 10), (3, 5, 11), (3, 5, 12), (3, 6, 9), (3, 6, 10), (3, 6, 11), (3, 6, 12), (3, 7, 9), (3, 7, 10), (3, 7, 11), (3, 7, 12), (3, 8, 9), (3, 8, 10), (3, 8, 11), (3, 8, 12), (4, 5, 9), (4, 5, 10), (4, 5, 11), (4, 5, 12), (4, 6, 9), (4, 6, 10), (4, 6, 11), (4, 6, 12), (4, 7, 9), (4, 7, 10), (4, 7, 11), (4, 7, 12), (4, 8, 9), (4, 8, 10), (4, 8, 11), (4, 8, 12)]


    """

    def __init__(self, *data):
        self._data_list = []
        for datum in data:
            self._data_list.append(datum)

    def add_data(self, *data):  # 添加生成笛卡尔积的数据列表
        for datum in data:
            self._data_list.append(datum)

    def build(self, return_list=False):
        """计算笛卡尔积"""
        import itertools
        if return_list:
            return [item for item in itertools.product(*self._data_list)]
        else:  # 返回set类型
            return {item for item in itertools.product(*self._data_list)}

# 另一种实现方法
# def fourier_odd(fx, x_tuple=(sympy.abc.x, 0, sympy.pi), T=0, silent=False):
#     x = x_tuple[0]
#     a = x_tuple[1]
#     b = x_tuple[2]


def fourier_odd(fx, x=sympy.abc.x, a=0, b=sympy.pi, T=0, silent=False):
    """
    计算奇延拓Fourier级数的Fourier系数,[0,b]区间,T周期(默认2b)
    """
    from sympy import pi
    n = sympy.Symbol('n', integer=True, positive=True)  # 定义符号n 为正整数

    if T == 0:
        T = 2 * b

    a0 = 0
    an = 0
    bn = sympy.simplify(sympy.integrate(fx * sympy.sin(2 * pi * n * x / T), (x, a, b)) * 4 / T)

    if not silent:
        print(f"{a0/2=}")
        print(f"{an=}")
        print(f"{bn=}")
    return [a0, an, bn]


def fourier_even(fx, x=sympy.abc.x, a=0, b=sympy.pi, T=0, silent=False):
    """计算偶延拓Fourier级数的Fourier系数,[0,b]区间,T周期(默认2b)"""
    from sympy import pi
    n = sympy.Symbol('n', integer=True, positive=True)  # 定义符号n 为正整数

    if T == 0:
        T = 2 * b

    a0 = sympy.simplify(sympy.integrate(fx, (x, a, b)) * 4 / T)
    an = sympy.simplify(sympy.integrate(fx * sympy.cos(2 * pi * n * x / T), (x, a, b)) * 4 / T)
    bn = 0

    if not silent:
        print(f"{a0/2=}")
        print(f"{an=}")
        print(f"{bn=}")
    return [a0, an, bn]


def fourier_series(fx, x=sympy.abc.x, a=-sympy.pi, b=sympy.pi, T=0, silent=False):
    """
    计算Fourier级数的Fourier系数,[a,b]区间,T周期(默认b-a)

    如果不加参数x=sympy.abc.x(即x=sympy.symbols('x')，没有限定条件)，则fx中其他的x会因与此函数中的x不是同一个x对象而导致计算错误

    注:sympy.fourier_series(1+x)或sympy.fourier_series(1+x, (x,-pi,pi))会得到
    FourierSeries(x + 1, (x, -pi, pi), (1, SeqFormula(Piecewise((2*sin(_n*pi)/_n, (_n > -oo) & (_n < oo) & Ne(_n, 0)), (2*pi, True))*cos(_n*x)/pi, (_n, 1, oo)), SeqFormula(Piecewise((-2*pi*cos(_n*pi)/_n + 2*sin(_n*pi)/_n**2, (_n > -oo) & (_n < oo) & Ne(_n, 0)), (0, True))*sin(_n*x)/pi, (_n, 1, oo))))
    这一串看不懂的东西，不如我这个好用

    """
    from sympy import pi
    n = sympy.Symbol('n', integer=True, positive=True)  # 定义符号n 为正整数

    if T == 0:
        T = b - a

    a0 = sympy.simplify(sympy.integrate(fx, (x, a, b)) * 2 / T)
    an = sympy.simplify(sympy.integrate(fx * sympy.cos(2 * pi * n * x / T), (x, a, b)) * 2 / T)
    bn = sympy.simplify(sympy.integrate(fx * sympy.sin(2 * pi * n * x / T), (x, a, b)) * 2 / T)

    if not silent:
        print(f"{a0/2=}")
        print(f"{an=}")
        print(f"{bn=}")
    return [a0, an, bn]


def generalized_fourier(fx, phi_nx, phi_0x=0, x=sympy.abc.x, a=-sympy.pi, b=sympy.pi, silent=False):
    """广义Fourier级数,phi_0x=0时不单独计算a0"""

    a0 = sympy.integrate(fx * phi_0x, (x, a, b))
    a0 = sympy.simplify(a0)

    an = sympy.integrate(fx * phi_nx, (x, a, b))
    an = sympy.simplify(an)

    if not silent:
        print(f"{a0=}")
        print(f"{an=}")
    return [a0, an]


def fourier_transform(fx, x=sympy.abc.x, silent=False):
    r"""Fourier变换

    .. math:: F(k) = \int_{-\infty}^\infty f(x) e^{- i x k} \mathrm{d} x.
    """
    x1, k = sympy.symbols('x,k', real=True)
    fx.subs(x, x1)  # 将fx中的 默认x 替换成 x1(实数x),若fx原来就是实数x也不会报错
    Fk = sympy.integrate(fx * sympy.exp(-sympy.I * k * x1), (x1, -oo, oo))
    if not silent:
        print(Fk)
    return Fk


def fourier_transform_inverse(Fk, k=sympy.abc.k, silent=False):
    r"""Fourier变换的逆变换

        .. math:: F(k) = \int_{-\infty}^\infty f(x) e^{- i x k} \mathrm{d} x.
        """
    x, k1 = sympy.symbols('x,k', real=True)
    Fk.subs(k, k1)  # 同理
    fx = sympy.integrate(Fk * sympy.exp(sympy.I * k1 * x), (k1, -oo, oo)) / (2 * sympy.pi)
    if not silent:
        print(fx)
    return fx


def legrendre(n):
    """勒让德多项式"""
    pnx = 1 / (2 ** n * sympy.factorial(n)) * sympy.diff((x * x - 1) ** n, x, n)
    # print(pnx)
    return pnx


def legrendre_list(n):
    """勒让德多项式表"""
    l1 = []
    for i in range(0, n):
        l1.append(legrendre(i))
    print(l1)  # [1, x, (3*x**2 - 1)/2, x*(5*x**2 - 3)/2, (8*x**4 + 24*x**2*(x**2 - 1) + 3*(x**2 - 1)**2)/8]
    return l1


def dot_product2(fx, gx, a=-1, b=1, silent=True):
    """积分形式的内积"""
    from sympy import integrate
    o1 = integrate(fx * gx, (x, a, b))
    if not silent:
        print(f"fx*gx={o1}")
    return o1


def schmidt_orthogonalization(fix: list, n, e: list, a=-1, b=1):
    """施密特正交化,输出为列表,n=len(fix),手动选择积分区间[a,b]"""
    from sympy import sqrt
    # dot_product=dot_product2 #另一种实现方法,手动选择内积

    if n == 1:
        e[0] = fix[0] / sqrt(dot_product2(fix[0], fix[0], a, b))
    else:
        schmidt_orthogonalization(fix, n - 1, e, a, b)
        e[n - 1] = fix[n - 1]
        for j in range(0, n - 1):
            e[n - 1] -= dot_product2(fix[n - 1], e[j], a, b) * e[j]
        e[n - 1] /= sqrt(dot_product2(e[n - 1], e[n - 1], a, b))


def schmidt(fix: list, n, e: list, dot_product=dot_product2):
    """施密特正交化,输出在e列表,n=len(fix),手动选择内积dot_product"""
    from sympy import sqrt

    if n == 1:
        e[0] = fix[0] / sqrt(dot_product(fix[0], fix[0]))
    else:
        schmidt(fix, n - 1, e)
        e[n - 1] = fix[n - 1]
        for j in range(0, n - 1):
            e[n - 1] -= dot_product(fix[n - 1], e[j]) * e[j]
        e[n - 1] /= sqrt(dot_product(e[n - 1], e[n - 1]))


def schmidt_orthogonalization_list(fix: list, n, a=-1, b=1, silent=True):
    """施密特正交化,输出为列表,n=len(fix),手动选择积分区间[a,b]"""
    e = []
    for i in range(0, n):
        e.append(None)
    schmidt_orthogonalization(fix, n, e, a, b)
    if not silent:
        print(e)
    return e


def schmidt_list(fix: list, n, dot_product=dot_product2, silent=True):
    """施密特正交化,输出为列表,n=len(fix),手动选择内积dot_product"""
    e = []
    for i in range(0, n):
        e.append(None)
    schmidt(fix, n, e, dot_product)
    if not silent:
        print(e)
    return e


def convolution1(ft, gx_t, silent=True):
    """卷积,核心算法"""
    from sympy import integrate
    t = sympy.symbols('t', real=True)
    o1 = integrate(ft * gx_t, (t, -oo, oo))
    if not silent:
        print(f"(f*g)(x)={o1}")
    return o1


def convolution(fx: str, gx: str):
    """卷积"""
    t = sympy.symbols('t', real=True)
    ft1 = fx.replace('x', 't')
    gx_t1 = gx.replace('x', '(x-t)')
    exec("ft2=" + ft1)
    exec("gx_t2=" + gx_t1)
    o1 = [None]  # 必须用引用类型(比如列表),否则报错
    exec("o1[0]=convolution1(ft2, gx_t2)")
    return o1[0]


def factorial(num, double=False, symbol=True):
    """阶乘或双阶乘"""
    if double:
        return sympy.factorial2(num)

    if symbol:
        return sympy.factorial(num)
    else:
        fa = 1
        for i in range(1, num + 1):
            fa *= i
        return fa


def average(values):
    """平均值

    :param values:Any
    :return:float

    Example:
        >>> print(average([20, 30, 10]))
        20.0
        >>>

    """
    return sum(values) / len(values)


def std(a, mode='n-1'):
    """样本标准差,默认除以n-1(无偏),除以n的是有偏的(相当于np.std(ddof=0))

    ddof - Means Delta Degrees of Freedom (n-自由度)

    pandas的std默认无偏。
    以下是numpy对其默认有偏的解释:

    The average squared deviation is normally calculated as
    ``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
    the divisor ``N - ddof`` is used instead. In standard statistical
    practice, ``ddof=1`` provides an unbiased estimator of the variance
    of the infinite population. ``ddof=0`` provides a maximum likelihood
    estimate of the variance for normally distributed variables. The
    standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an
    unbiased estimate of the standard deviation per se.

    Example:
        >>> print(std([20, 30, 10]))
        10.0000000000000
        >>> print(std([1200, 3299.5, 2133, 5433, 3299.5, 4432]))
        1523.35163373398
        >>> print(std([1200, 3299.5, 2133, 5433, 3299.5, 4432],'n'))
        1390.62342134742
        >>>
    """
    std1 = 0
    ave1 = average(a)
    for v1 in a:
        std1 += (v1 - ave1) ** 2
    if mode == 'n-1':
        std1 /= (len(a) - 1)
    elif mode == 'n':
        std1 /= len(a)
    else:
        return None
    return math.sqrt(std1)


class Prime(object):
    """素数类"""

    def __init__(self,n=32):
        if n<=2:
            self.primes = [2,3]
        else:
            self.primes = Prime.prime_n(n)  # 前n个素数

    @staticmethod
    def is_prime(n):
        """
        判断一个数是否为素数

        # 100以内的素数
        for i in range(100):
            if is_prime(i):
                print(i)
        """
        if n < 2:
            return False
        if n == 2:
            return True
        elif n % 2 == 0:
            return False

        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_ver2(self, n):
        """
        判断一个数是否为素数 ver2 只检查素因子,更快
        """
        if n < 2:
            return False
        if n == 2:
            return True
        elif n % 2 == 0:
            return False

        for m in self.primes:
            if n % m == 0:#包含了m**2==n
                return False
            if m**2>n:
                return True

        m=self.primes[-1]#append_next_prime内调用is_prime_ver2必有m**2>n,不会执行到这一行，不会导致循环递归
        while m**2<n:#不可能等于，否则return False
            m=self.append_next_prime()
            if n % m == 0:#包含了m**2==n
                return False
        return True

    def append_next_prime(self):
        k = self.primes[-1]+2#k至少是3+2=5
        while not self.is_prime_ver2(k):
            k+=2
        self.primes.append(k)
        return k

    def prime_front_n(self, n):
        """前n个素数"""
        while n>len(self.primes):
            self.append_next_prime()
        return self.primes[:n]

    @staticmethod
    def prime_n(n):
        """前n个素数"""

        q = [2]
        # j = 3
        for i in range(1, n):
            # while not Prime.is_prime(j):
            #     j += 2
            # q.append(j)
            # j += 2
            q.append(Prime.next_prime(q[-1]))

        # print(q)
        return q

    @staticmethod
    def prime_the_n(n):
        """第n个素数"""
        return Prime.prime_n(n)[n - 1]

    @staticmethod
    def next_prime(n):
        """下一个素数(从n+1开始找)"""
        if n < 2:
            return 2

        if n % 2 == 0:
            m = n + 1
        else:
            m = n + 2
        while not Prime.is_prime(m):
            m += 2
        return m

    @staticmethod
    def prime_between(start,end):
        """[start,end)间的所有素数"""
        q=[]
        m=start-1
        while m+1<end:
            m=Prime.next_prime(m)
            q.append(m)

        return q

    def __next__(self):
        self.primes.append(Prime.next_prime(self.primes[-1]))
        return self

    def __iter__(self):
        return self


def fact(n):
    """
    整数因式分解

    # from sympy.ntheory import factorint
    # ntheory指 数论
    # 相当于sympy.factorint(n)
    """
    factors={}
    rest=abs(n)
    m=2
    while rest>1:#rest余1时分解完成
        if rest%m==0:
            rest//=m
            if m in factors:
                factors[m]+=1
            else:
                factors[m]=1
        else:
            m=Prime.next_prime(m)
    return factors


def eular(n):
    fact_n=fact(n)
    # # 法一，书上的公式。只涉及整数，结果必定精确
    # m=1
    # for k,v in fact_n.items():
    #    m*=k**(fact_n[k]-1)*(k-1)

    # # 法二，化简公式，有浮点数，不保证精确，不适用于大整数
    # m=n
    # for k in fact_n.keys():
    #     m*=(1-1/k)
    # m=int(m)

    # 法三，结合法一和法二，比较好
    m=n
    for k in fact_n.keys():
        m=m//k * (k-1)

    return m


def gcd(*v):
    if len(v)<=1:
        raise ValueError("gcd() require at least two arguments")

    def _gcd(a, b):
        """辅助函数"""
        if a == 0 and b == 0:
            raise ValueError("gcd(0,0) not defined")
        if b == 0:
            return abs(a)
        # if a == 0:#a==0可按下面的程序执行，结果一样
        #     return abs(b)

        b = abs(b)
        yu = a % b
        while yu > 0:
            a = b
            b = yu
            yu = a % b
        return b

    g=v[0]
    for i in range(1,len(v)):
        # 可以用functool.reduce()，这里手动实现
        g=_gcd(g,v[i])
    return g


def lcm(*v):
    if len(v)<=1:
        raise ValueError("lcm() require at least two arguments")

    def _lcm(a, b):
        if a == 0 or b == 0:
            raise ValueError("lcm(0,x) or lcm(x,0) not defined")
        return abs(a*b)//gcd(a,b)

    g = v[0]
    for i in range(1, len(v)):
        # 可以用functool.reduce()，这里手动实现
        g = _lcm(g, v[i])
    return g


def sgn(n):
    if n>0:
        return 1
    elif n<0:
        return -1
    else:
        return 0


def euclidean_algorithm(a,b,silent=True):
    """
    扩展欧几里得算法，非递归实现
    返回gcd(a,b)以及ax+by=gcd(a,b)的(x,y)

    :return :gcd(a,b),x,y
    """
    if a==0 and b==0:
        raise ValueError("gcd(0,0) not defined")
    if b==0:
        if not silent:
            print(f"{a}*{sgn(a)} + {b}*t = {abs(a)}")
        return abs(a),1,0
    # if a == 0:#a==0可按下面的程序执行，结果一样
    #     if not silent:
    #         print(f"{a}*t + {b}*{sgn(b)} = {abs(b)}")
    #     return abs(b), 0, 1

    origin_a = a
    origin_b = b
    b=abs(b)
    q=a//b#商
    r=a%b#余数
    quotients = [q]
    # remainders = [a,b,r]
    while r>0:
        a=b
        b=r
        r=a%b
        quotients.append(a//b)
        # remainders.append(r)

    g=b

    # # 这样要多循环一次
    # i_q = len(quotients)
    # x, y = 1,0

    i_q=len(quotients)-1
    x, y = 0,1
    # 下次循环时 x, y = 1,-quotients[i_q-1]
    # 但有可能i_q==0，quotients[i_q-1]==quotients[-1]==quotients[0]将导致逻辑错误，因此放到循环里执行，而不是直接按上式赋初值

    while i_q>=1:
        x, y = y, x - quotients[i_q-1] * y
        i_q -= 1

    y = sgn(origin_b) * y
    if not silent:
        print(f"{origin_a}*({x}-{origin_b}t) + {origin_b}*({y}+{origin_a}t) = {g}")
    return g,x,y


def euclidean_algorithm_recursion(a,b,silent=True):
    """
    扩展欧几里得算法，递归实现
    返回gcd(a,b)以及ax+by=gcd(a,b)的(x,y)

    :return :gcd(a,b),x,y
    """
    if b==0:
        if not silent:
            print(f"{a}*{sgn(a)} + {b}*t = {abs(a)}")
        return abs(a),1,0
    else:
        origin_b=b
        b=abs(b)

        def _euclidean_algorithm_extended(a, b):
            """
            扩展欧几里得算法递归辅助函数，a,b>0
            返回gcd(a,b)以及ax+by=gcd(a,b)的(x,y)

            :return :gcd(a,b),x,y
            """
            if b == 0:
                return a, 1, 0
            else:
                g, x, y = _euclidean_algorithm_extended(b, a % b)
                # g=gcd(a,b)=gcd(b,a%b)
                x, y = y, x - a // b * y
                return g, x, y

        g,x,y=_euclidean_algorithm_extended(a,b)
        y=sgn(origin_b)*y
        if not silent:
            print(f"{a}*({x}-{origin_b}t) + {origin_b}*({y}+{a}t) = {g}")
        return g,x,y


def mod_m_inverse(a,m):
    """
    设m>0，gcd(a,m)=1，求b使[a][b]=[1]，即在乘法群中[b]是[a]的逆元

    返回b的主值(即 0~m-1 之间的b)
    """
    if m<=0:
        raise ValueError("m<=0")
    g,b,n=euclidean_algorithm(a,m)
    b=b%m  # 计算机中a%b==a-a//b*b，m>0时一定返回0~m-1之间的数
    return b


def solve_congruence(a,b,m,silent=True):
    """
    同余方程 ax≡b mod m 求解
    若有解 x mod (m//gcd(a,m)) 则返回(x, m//gcd(a,m));若无解则返回None
    """
    d=gcd(a,m)
    if b%d!=0:
        if not silent:
            print(f"{a}x ≡ {b} mod {m} 无解")
        return None
    else:
        c=mod_m_inverse(a//d,m//d)
        x=b//d*c
        x=x%(m//d)
        if not silent:
            print(f"{a}x ≡ {b} mod {m} 解集为 {x} mod {m//d}")
        return x,m//d


def change_base(n,k1=10,k2=2):
    """
    整数进制(base，或称radix(基数))转换
    k1进制n转为k2进制

    k1==0 or 2<=k1<=36

    2<=k2<=36

    以下是int的文档解释
    The base defaults to 10. Valid bases are 0 and 2-36. Base 0 means to interpret the base from the string as an integer literal.
     >>> int('0b100', base=0)
     4
    """
    if not (k1==0 or 2<=k1<=36):
        raise ValueError(f"error base {k1}")
    if not (2<=k2<=36):
        raise ValueError(f"error base {k2}")

    def dec_to_other(d,other):
        if other==10:
            return d
        q=[]
        while d>0:
            q.append(d%other)
            d//=other

        s=''
        l=len(q)
        for i in q[::-1]:
            if 0<=i<=9:
                s+=f'{i}'
            else:
                s+=chr(ord('a')+i-10)
        return s

    if k1!=10:
        dec_n=int(n,k1)#n的10进制
        if k2==10:
            return dec_n
        else:
            return dec_to_other(dec_n,k2)
    else:
        return dec_to_other(int(n), k2)


def pow_n_mod_m(a,n,m):
    """
    a**n mod m
    即 a**n % m，其实当a很大n很小时，python的a**n % m计算很快，但n很大时计算非常慢
    输出主值(0~m-1)
    """
    b=list(map(int,bin(n)[2:]))#去掉'0b',第一个是nk
    # b=bin(n)[-1:1:-1]#去掉'0b'并且倒转

    a0=a**b[0] % m
    k=len(b)-1
    if k==0:
        return a0
    else:
        for i in range(1,k+1):
            a0=a0**2 * a**b[i] % m
        return a0


def solve_congruence_set(a:list,b:list,m:list,silent=True):
    """
    同余方程组求解

    .. math::
        a_i x ≡ b_i mod m_i

    若有解 x0 mod m0 则返回(x0, m0);若无解则返回None
    """
    k=min([len(a),len(b),len(m)])
    x=[]
    new_m=[]
    for i in range(k):
        d_i=gcd(a[i],m[i])
        if b[i]%d_i!=0:
            if not silent:
                print(f"{a}x ≡ {b} mod {m} 无解")
            return None
        else:
            x_i,m_i=solve_congruence(a[i],b[i],m[i])
            x.append(x_i)
            new_m.append(m_i)

    def solve_2(x_i,x_j,m_i,m_j):
        """解两个方程的方程组，返回解 x≡x_ij mod m_ij 或 None(无解)"""
        m_ij,k_i,k_j=euclidean_algorithm(m_i,m_j)
        k_i=-k_i
        c_i=m_i//m_ij
        # c_j=m_j//m_ij
        if (x_i-x_j)%m_ij!=0:
            return None
        else:
            return (x_i-x_j)*k_i*c_i+x_i, m_i*m_j//m_ij  # 第二项即lcm(m_i,m_j)

    x0, m0 = x[0], new_m[0]
    for i in range(1,k):
        # 可以用functool.reduce()，这里手动实现
        tmp=solve_2(x0,x[i],m0,new_m[i])
        if tmp:
            x0, m0 = tmp[0], tmp[1]
        else:
            if not silent:
                print(f"{a}x ≡ {b} mod {m} 无解")
            return None

    x0=x0%m0  # 取0~m0-1之间的主值
    if not silent:
        print(f"{a}x ≡ {b} mod {m} 解集为 {x0} mod {m0}")
    return x0, m0


def primality_test(n,algorithm="Fermat",test_time=10):
    """
    素性判定
    判断n是否为素数
    """
    if n<=1:
        return False
    if n<=1009:
        front_1000=Prime.prime_between(2,1010)
        if n in front_1000:
            return True
        else:
            return False

    if algorithm=="Fermat":
        # 费马伪素数
        a=Prime.prime_between(2,1010)
        for i in range(test_time):
            # if random.choice(a)**(n-1)%n!=1:#很慢
            if pow_n_mod_m(random.choice(a),n-1,n)!=1:
                return False
        print(f'{n} 可能是素数,经过{test_time}次Fermat测试')
        return False  # 此处False表示未能判定

    if algorithm=="miller labin":
        # 米勒-拉宾 伪素数
        if n%2==1:
            k=1#注意从1开始，因为循环之前n已经除以2了
            n2=n//2
            while n2%2==0:
                k+=1
                n2//=2

            # 此时n==n2 * 2**k
            a_list = Prime.prime_between(2, 1010)

            for i in range(0,test_time):
                # a=random.randint(2,99999)
                a=random.choice(a_list)

                condition1=False
                condition2=True

                if pow_n_mod_m(a,n2,n)!=1:
                    condition1=True

                # 这个循环可以优化一下，见pylearn\miller_labin.txt，但优化后的代码不易理解
                for j in range(0,k):
                    if pow_n_mod_m(a, 2 ** j * n2, n) == n - 1:
                        condition2=False

                if condition1 and condition2:
                    return False

            print(f'{n} 可能是素数,经过{test_time}次miller labin测试')
            return False  # 此处False表示未能判定

        else:
            return False


def primitive_root(m):
    """
    求模m的一个原根(最小正原根)
    m=2,4,p**k,2*p**k 其中','表示或
    """
    if m==2:
        return 1
    elif m==4:
        return 3

    fac=fact(m)
    p_s=list(fac.keys())

    def multiple_order(a,p):
        """p是素数，2<=a<=p-1，计算乘法群(Z/pZ)×中a模p的阶"""
        for i in range(1,p):#阶是正整数，因此要从1开始循环
            if pow_n_mod_m(a,i,p)==1:
                return i

    def mod_p_primitive_root(p):
        for a in range(2, p):
            if multiple_order(a, p) == p - 1:
                return a

    if len(p_s)==1 and p_s[0]>2:
        p=p_s[0]
        g=mod_p_primitive_root(p)
        if pow_n_mod_m(g,p-1,p**2)%p**2!=1:
            return g
        else:
            return g+p

    if len(p_s)>=2:
        if 2 in p_s and fac[2]==1:
            p=0
            for i in p_s:
                if i!=2:
                    p=i
            g2 = primitive_root(m//2)
            if g2%2:#奇数
                return g2
            else:
                return g2+m//2

    raise ValueError("m != 2 or 4 or p**k or 2*p**k")


def discrete_logarithm(g,a,n):
    """
    离散对数，n为循环群G的阶(G同构于Z/nZ(加法))，g为G的生成元，a∈G(一般a也是个集合(同余类))
    [k]=log_g(a)是一个模n的同余类
    返回0~n-1之间的[k]的主值k

    .. math::
        log_g(a)
    """
    # k=log_g(a),a**k=g,k至多=n-1
    a_main=a%n
    for k in range(0,n):
        if g*k%n==a_main:
            return k


def legendre_symbol(a,p):
    """
    勒让德符号(a/p)
    由欧拉判别法知，(a/p) mod p = a**((p-1)/2) mod p
    返回0,-1,1之一
    """
    r=pow_n_mod_m(a,(p-1)//2,p)
    if r==p-1:
        return -1
    else:  # 0或1，不可能是其他值
        return r


def quick_pow(self, a, b):
    """快速幂 a**b"""
    r=1
    base=a
    while b!=0:
        if b%2!=0:
            r*=base
        base*=base
        b>>=1

    return r


def run():
    import time
    import doctest

    start_time = time.time()
    # fourier_even(1 + x)
    # fourier_odd(1 + x)
    # fourier_series(1 + x, 0, sympy.pi)
    # 输出结果中Piecewise表示分段函数,Ne(n,0)表示n!=0,True表示除去前面(Ne(n,0))的情况(即n==0)
    # 已修复n的类型,不会输出上述结果了

    # print(std([20,30,10]))
    # print(std([1200, 3299.5, 2133, 5433, 3299.5, 4432]))
    # print(std([1200, 3299.5, 2133, 5433, 3299.5, 4432],'n'))

    doctest.testmod()

    print("运行时间: {} s".format(time.time() - start_time))


if __name__ == '__main__':
    run()
