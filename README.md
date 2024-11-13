# diskevolution
Evolution code for a viscously evolving gaseous disk.


Throughout our work, following the work of Guilera & Sándor (2017), we used a full implicit Crank-Nicholson method to evolve the density profile of a 1D gaseous disk, solving the viscous evolution equation considering zero torques at the boundaries.

$m_9$

\begin{eqnarray}\label{eq:viscosity}
    \frac{\partial\Sigma_{\rm gas}}{\partial t} &=& \frac{1}{r} \frac{\partial}{\partial r} \left[ 3r^{1/2} \frac{\partial}{\partial r}(\nu \Sigma_{\rm gas} r^{1/2}) \right] ,
 \\
    \nu &=& \alpha c_s H   , 
\end{eqnarray}


The file mymodule.py contains the module and macro.py contains an example of the code being used.

For any questions, contact me at mbest@uc.cl
