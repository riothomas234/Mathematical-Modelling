<hr style="border:1px solid black"> </hr>

## Section 2. Submitting Your Work for Course Credit

### Fitting a set of data to a periodic function with a Fourier expansion


We will use matrix inversion to fit a set of data to a periodic function. 

#### Fitting by matrix inversion 

We want a set of data fitted by a function of the general form

\begin{equation*}
f(x) = \sum_k a_k g_k(x)
\end{equation*}

where each $g_k(x)$ is a predefined (normally simple) function of $x$, 
and the coefficients $a_k$ need to be determined by fitting to the data.

In the case where we can describe a set of data by such $f(x)$, 
and **where the number of data points is equal to the number of terms**, 
we have a well-defined set of simultaneous equations of the form

\begin{equation*}
\begin{array}{c}
 f(x_1 ) = a_1 g_1 (x_1 ) + a_2 g_2 (x_1 ) +  \cdots  + a_n g_n (x_1 ) \\ 
 f(x_2 ) = a_1 g_1 (x_2 ) + a_2 g_2 (x_2 ) +  \cdots  + a_n g_n (x_2 ) \\ 
  \cdots  \\ 
 f(x_n ) = a_1 g_1 (x_n ) + a_2 g_2 (x_n ) +  \cdots  + a_n g_n (x_n ) \\ 
 \end{array}
\end{equation*}

in which everything is known except the coefficients $a_k$.
This can be expressed in matrix form:

\begin{equation}
\left( {\begin{array}{*{20}c}
   {f(x_1 )}  \\
   {f(x_2 )}  \\
    \vdots   \\
   {f(x_n )}  \\
\end{array}} \right) = \left( {\begin{array}{*{20}c}
   {g_1 (x_1 )} & {g_2 (x_1 )} &  \cdots  & {g_n (x_1 )}  \\
   {g_1 (x_2 )} & {g_2 (x_2 )} &  \cdots  & {g_n (x_2 )}  \\
    \vdots  &  \vdots  &  \ddots  &  \vdots   \\
   {g_1 (x_n )} & {g_2 (x_n )} &  \cdots  & {g_n (x_n )}  \\
\end{array}} \right) \times \left( {\begin{array}{*{20}c}
   {a_1 }  \\
   {a_2 }  \\
    \vdots   \\
   {a_n }  \\
\end{array}} \right)
\end{equation}

which can be written as

\begin{equation}
{\bf{f}} = {\bf{g}} \times {\bf{a}}
\end{equation}

Our aim is to find the coefficients in the column matrix $\mathbf{a}$, which are obtained from the matrix equation

\begin{equation}
{\bf{a}} = {\bf{g}}^{ - 1}  \times {\bf{f}}
\end{equation}

#### Fourier series for fitting a periodic function

  A periodic function $f(x)$ of period $a$, such that $f(x + N a)= f(x)$ for
any integer $N$, can be expanded using a so-called Fourier series

\begin{equation*}
\label{fourier}
f(x) = b_0 + \sum_{n=1}^{\infty} [ b_n \sin( n k x) + c_n \cos(n k x) ]
\end{equation*}
where $k = 2\pi/a$.

If we know the value of a periodic function in a set of points within a period,
we can obtain the smoothest periodic interpolation of those
points (such that the function does go through them) by truncating the sum
to the lowest values of $n$ needed, such that the number of variables coincides
with the number of data points for $f$.

Let us consider for instance 7 points, which will mean truncating the series
at $n=3$. The fitting function $f(x)$ is therefore

\begin{equation*}
f(x) = a_1 g_1(x) + a_2 g_2(x) + a_3 g_3(x) + \ldots  + a_7 g_7(x)
\end{equation*}

with the linear coefficients to be fitted

\begin{equation*}
a_1 = b_0 \; , \; a_2 = b_1 \, ; \; a_3 = c_1 \, ; \; a_4 = b_2 \, ; \; a_5 = c_2 \, ; \;
a_6 = b_3 \, ; \; a_7 = c_3 \, .
\end{equation*}

and the $g$ functions 

\begin{align*}
g_1(x) &= 1 \\
g_2(x) &= \sin( k x) \\
g_3(x) &= \cos(k x) \\
g_4(x) &= \sin( 2k x) \\
g_5(x) &= \cos(2k x) \\
g_6(x) &= \sin( 3k x) \\
g_7(x) &= \cos(3k x) 
\end{align*}

