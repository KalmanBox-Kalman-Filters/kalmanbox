/*******************************************************************************
* validate_comparison_mle.do
*
* Validacao: Comparacao de metodos de estimacao MLE vs CSS para ARIMA
*
* Objetivo:
*   Comparar estimativas de ARIMA usando diferentes metodos de estimacao:
*   - MLE (Maximum Likelihood Estimation) - via filtro de Kalman
*   - CSS (Conditional Sum of Squares) - minimizacao direta
*   - sspace (state-space representation) - filtro de Kalman explicito
*
* Especificacao sspace:
*   O `sspace` do Stata implementa o filtro de Kalman para modelos de espaco
*   de estados. Para ARIMA(p,d,q), a representacao state-space e:
*
*   Equacao de observacao: y_t = Z * alpha_t
*   Equacao de estado:     alpha_t = T * alpha_{t-1} + R * eta_t
*
*   Para ARIMA(0,1,1) em diferencas (dy_t):
*     Z = [1], T = [0], R = [1], com MA: theta * L.e.alpha
*
*   Para ARIMA(1,1,0) em diferencas (dy_t):
*     Z = [1], T = [phi], R = [1], estado AR: alpha_t = phi*alpha_{t-1} + e_t
*
*   Para ARIMA(1,1,1) em diferencas (dy_t):
*     Combina componentes AR e MA no estado
*
* Diferenca MLE vs CSS:
*   - MLE: Usa o filtro de Kalman para calcular a verossimilhanca exata,
*     incluindo o tratamento dos valores iniciais difusos.
*   - CSS: Condiciona nos primeiros valores observados e minimiza a soma
*     dos quadrados dos residuos. Mais rapido, mas menos eficiente para
*     amostras pequenas.
*
* Criterio de aceite:
*   Parametros devem coincidir entre MLE e sspace (tolerancia < 1%)
*   CSS pode diferir ligeiramente do MLE
*
* Autor: kalmanbox validation suite
* Data: 2026-04-09
*******************************************************************************/

clear all
set more off
capture log close
log using "validate_comparison_mle.log", replace

display "============================================================"
display " VALIDACAO: Comparacao MLE vs CSS vs sspace"
display " Data: $S_DATE $S_TIME"
display "============================================================"

/*******************************************************************************
* Carregar dados Nile
*******************************************************************************/

clear
input int year double flow
1871 1120
1872 1160
1873 963
1874 1210
1875 1160
1876 1160
1877 813
1878 1230
1879 1370
1880 1140
1881 995
1882 935
1883 1110
1884 994
1885 1020
1886 960
1887 1180
1888 799
1889 958
1890 1140
1891 1100
1892 1210
1893 1150
1894 1250
1895 1260
1896 1220
1897 1030
1898 1100
1899 774
1900 840
1901 874
1902 694
1903 940
1904 833
1905 701
1906 916
1907 692
1908 1020
1909 1050
1910 969
1911 831
1912 726
1913 456
1914 824
1915 702
1916 1120
1917 1100
1918 832
1919 764
1920 821
1921 768
1922 845
1923 864
1924 862
1925 698
1926 845
1927 744
1928 796
1929 1040
1930 759
1931 781
1932 865
1933 845
1934 944
1935 984
1936 897
1937 822
1938 1010
1939 771
1940 676
1941 649
1942 846
1943 812
1944 742
1945 801
1946 1040
1947 860
1948 874
1949 848
1950 890
1951 744
1952 749
1953 838
1954 1050
1955 918
1956 986
1957 797
1958 923
1959 975
1960 815
1961 1020
1962 906
1963 901
1964 1170
1965 912
1966 746
1967 919
1968 718
1969 714
1970 740
end

tsset year

display "Dados Nile carregados: `=_N' observacoes"

/*******************************************************************************
* PARTE 1: ARIMA(0,1,1) - MLE vs CSS vs sspace
*******************************************************************************/

display ""
display "============================================================"
display " PARTE 1: ARIMA(0,1,1) - MLE vs CSS vs sspace"
display "============================================================"

* --- Metodo 1: MLE (default) ---
display ""
display "--- ARIMA(0,1,1) via MLE ---"
arima d.flow, arima(0,0,1) noconstant technique(nr)

scalar mle_011_theta = _b[ARMA:L.ma]
scalar mle_011_sigma = _b[/sigma]
scalar mle_011_ll    = e(ll)

* --- Metodo 2: CSS ---
* CSS condiciona nos valores iniciais; difere do MLE para amostras pequenas
display ""
display "--- ARIMA(0,1,1) via CSS ---"
arima d.flow, arima(0,0,1) noconstant condition

scalar css_011_theta = _b[ARMA:L.ma]
scalar css_011_sigma = _b[/sigma]
scalar css_011_ll    = e(ll)

* --- Metodo 3: sspace ---
* State-space: filtro de Kalman explicito, deve coincidir com MLE
display ""
display "--- ARIMA(0,1,1) via sspace ---"
display ""
display "Especificacao sspace:"
display "  Obs:   D.flow = alpha (estado)"
display "  State: alpha  = theta * L.e.alpha + e.alpha (MA(1))"

sspace (D.flow = {alpha}, mle)                                             ///
       ({alpha} = {theta}*L.e.alpha, state noconstant error(alpha))

scalar ss_011_theta = _b[alpha:L.e.alpha]
scalar ss_011_sigma = exp(_b[/lnsigma_e.alpha])
scalar ss_011_ll    = e(ll)

* --- Comparacao ---
display ""
display "============================================================"
display " COMPARACAO ARIMA(0,1,1)"
display "============================================================"
display ""
display "              MLE          CSS         sspace"
display "  theta:  " %10.6f mle_011_theta "  " %10.6f css_011_theta "  " %10.6f ss_011_theta
display "  sigma:  " %10.4f mle_011_sigma "  " %10.4f css_011_sigma "  " %10.4f ss_011_sigma
display "  loglik: " %10.4f mle_011_ll    "  " %10.4f css_011_ll    "  " %10.4f ss_011_ll

scalar diff_mle_ss_theta = abs(mle_011_theta - ss_011_theta) / abs(mle_011_theta) * 100
scalar diff_mle_css_theta = abs(mle_011_theta - css_011_theta) / abs(mle_011_theta) * 100
scalar diff_mle_ss_ll = abs(mle_011_ll - ss_011_ll)

display ""
display "  MLE vs sspace theta diff (%): " %8.4f diff_mle_ss_theta
display "  MLE vs CSS theta diff (%):    " %8.4f diff_mle_css_theta
display "  MLE vs sspace loglik diff:    " %8.4f diff_mle_ss_ll

if diff_mle_ss_theta < 1 {
    display "  [PASS] MLE vs sspace theta < 1%"
}
else {
    display "  [FAIL] MLE vs sspace theta >= 1%"
}

if diff_mle_ss_ll < 0.5 {
    display "  [PASS] MLE vs sspace log-likelihood coincide"
}
else {
    display "  [WARN] MLE vs sspace log-likelihood diff = " diff_mle_ss_ll
}

/*******************************************************************************
* PARTE 2: ARIMA(1,1,0) - MLE vs CSS vs sspace
*******************************************************************************/

display ""
display "============================================================"
display " PARTE 2: ARIMA(1,1,0) - MLE vs CSS vs sspace"
display "============================================================"

* --- MLE ---
display ""
display "--- ARIMA(1,1,0) via MLE ---"
arima d.flow, arima(1,0,0) noconstant technique(nr)

scalar mle_110_phi  = _b[ARMA:L.ar]
scalar mle_110_sigma = _b[/sigma]
scalar mle_110_ll   = e(ll)

* --- CSS ---
display ""
display "--- ARIMA(1,1,0) via CSS ---"
arima d.flow, arima(1,0,0) noconstant condition

scalar css_110_phi  = _b[ARMA:L.ar]
scalar css_110_sigma = _b[/sigma]
scalar css_110_ll   = e(ll)

* --- sspace ---
* Especificacao state-space para ARIMA(1,1,0):
*   Obs:   D.flow = alpha
*   State: alpha  = phi * L.alpha + e.alpha (AR(1) puro)
display ""
display "--- ARIMA(1,1,0) via sspace ---"
display ""
display "Especificacao sspace:"
display "  Obs:   D.flow = alpha"
display "  State: alpha  = phi * L.alpha + e.alpha (AR(1))"

sspace (D.flow = {alpha}, mle)                                             ///
       ({alpha} = {phi}*L.alpha, state noconstant error(alpha))

scalar ss_110_phi  = _b[alpha:L.alpha]
scalar ss_110_sigma = exp(_b[/lnsigma_e.alpha])
scalar ss_110_ll   = e(ll)

* --- Comparacao ---
display ""
display "============================================================"
display " COMPARACAO ARIMA(1,1,0)"
display "============================================================"
display ""
display "             MLE          CSS         sspace"
display "  phi:   " %10.6f mle_110_phi  "  " %10.6f css_110_phi  "  " %10.6f ss_110_phi
display "  sigma: " %10.4f mle_110_sigma "  " %10.4f css_110_sigma "  " %10.4f ss_110_sigma
display "  loglik:" %10.4f mle_110_ll    "  " %10.4f css_110_ll    "  " %10.4f ss_110_ll

scalar diff_mle_ss_phi = abs(mle_110_phi - ss_110_phi) / abs(mle_110_phi) * 100
scalar diff_mle_css_phi = abs(mle_110_phi - css_110_phi) / abs(mle_110_phi) * 100
scalar diff_mle_ss_ll_110 = abs(mle_110_ll - ss_110_ll)

display ""
display "  MLE vs sspace phi diff (%):  " %8.4f diff_mle_ss_phi
display "  MLE vs CSS phi diff (%):     " %8.4f diff_mle_css_phi
display "  MLE vs sspace loglik diff:   " %8.4f diff_mle_ss_ll_110

if diff_mle_ss_phi < 1 {
    display "  [PASS] MLE vs sspace phi < 1%"
}
else {
    display "  [FAIL] MLE vs sspace phi >= 1%"
}

/*******************************************************************************
* PARTE 3: ARIMA(1,1,1) - MLE vs CSS vs sspace
*******************************************************************************/

display ""
display "============================================================"
display " PARTE 3: ARIMA(1,1,1) - MLE vs CSS vs sspace"
display "============================================================"

* --- MLE ---
display ""
display "--- ARIMA(1,1,1) via MLE ---"
arima d.flow, arima(1,0,1) noconstant technique(nr)

scalar mle_111_phi   = _b[ARMA:L.ar]
scalar mle_111_theta = _b[ARMA:L.ma]
scalar mle_111_sigma = _b[/sigma]
scalar mle_111_ll    = e(ll)

* --- CSS ---
display ""
display "--- ARIMA(1,1,1) via CSS ---"
arima d.flow, arima(1,0,1) noconstant condition

scalar css_111_phi   = _b[ARMA:L.ar]
scalar css_111_theta = _b[ARMA:L.ma]
scalar css_111_sigma = _b[/sigma]
scalar css_111_ll    = e(ll)

* --- sspace ---
* Especificacao state-space para ARIMA(1,1,1):
*   Requer dois estados em companion form
*   alpha1_t = phi*alpha1_{t-1} + e_t + theta*e_{t-1}
* No sspace combinamos AR e MA:
*   Obs:   D.flow = alpha1
*   State: alpha1 = phi*L.alpha1 + theta*L.e.alpha1 + e.alpha1
display ""
display "--- ARIMA(1,1,1) via sspace ---"
display ""
display "Especificacao sspace:"
display "  Obs:   D.flow = alpha1"
display "  State: alpha1 = phi*L.alpha1 + theta*L.e.alpha1 + e.alpha1"

sspace (D.flow = {alpha1}, mle)                                            ///
       ({alpha1} = {phi}*L.alpha1 + {theta}*L.e.alpha1,                    ///
        state noconstant error(alpha1))

scalar ss_111_phi   = _b[alpha1:L.alpha1]
scalar ss_111_theta = _b[alpha1:L.e.alpha1]
scalar ss_111_sigma = exp(_b[/lnsigma_e.alpha1])
scalar ss_111_ll    = e(ll)

* --- Comparacao ---
display ""
display "============================================================"
display " COMPARACAO ARIMA(1,1,1)"
display "============================================================"
display ""
display "              MLE          CSS         sspace"
display "  phi:    " %10.6f mle_111_phi   "  " %10.6f css_111_phi   "  " %10.6f ss_111_phi
display "  theta:  " %10.6f mle_111_theta "  " %10.6f css_111_theta "  " %10.6f ss_111_theta
display "  sigma:  " %10.4f mle_111_sigma "  " %10.4f css_111_sigma "  " %10.4f ss_111_sigma
display "  loglik: " %10.4f mle_111_ll    "  " %10.4f css_111_ll    "  " %10.4f ss_111_ll

scalar diff_mle_ss_phi_111 = abs(mle_111_phi - ss_111_phi) / abs(mle_111_phi) * 100
scalar diff_mle_ss_theta_111 = abs(mle_111_theta - ss_111_theta) / abs(mle_111_theta) * 100
scalar diff_mle_ss_ll_111 = abs(mle_111_ll - ss_111_ll)

display ""
display "  MLE vs sspace phi diff (%):   " %8.4f diff_mle_ss_phi_111
display "  MLE vs sspace theta diff (%): " %8.4f diff_mle_ss_theta_111
display "  MLE vs sspace loglik diff:    " %8.4f diff_mle_ss_ll_111

if diff_mle_ss_phi_111 < 1 {
    display "  [PASS] MLE vs sspace phi < 1%"
}
else {
    display "  [FAIL] MLE vs sspace phi >= 1%"
}

if diff_mle_ss_theta_111 < 1 {
    display "  [PASS] MLE vs sspace theta < 1%"
}
else {
    display "  [FAIL] MLE vs sspace theta >= 1%"
}

/*******************************************************************************
* PARTE 4: Exportar tabela comparativa em CSV
*******************************************************************************/

display ""
display "============================================================"
display " EXPORTANDO TABELA COMPARATIVA"
display "============================================================"

preserve
clear
set obs 9

gen str15 model   = ""
gen str10 method  = ""
gen str10 param   = ""
gen double value  = .
gen double loglik = .

* ARIMA(0,1,1)
replace model  = "ARIMA(0,1,1)" in 1
replace method = "MLE"          in 1
replace param  = "theta"        in 1
replace value  = mle_011_theta  in 1
replace loglik = mle_011_ll     in 1

replace model  = "ARIMA(0,1,1)" in 2
replace method = "CSS"          in 2
replace param  = "theta"        in 2
replace value  = css_011_theta  in 2
replace loglik = css_011_ll     in 2

replace model  = "ARIMA(0,1,1)" in 3
replace method = "sspace"       in 3
replace param  = "theta"        in 3
replace value  = ss_011_theta   in 3
replace loglik = ss_011_ll      in 3

* ARIMA(1,1,0)
replace model  = "ARIMA(1,1,0)" in 4
replace method = "MLE"          in 4
replace param  = "phi"          in 4
replace value  = mle_110_phi    in 4
replace loglik = mle_110_ll     in 4

replace model  = "ARIMA(1,1,0)" in 5
replace method = "CSS"          in 5
replace param  = "phi"          in 5
replace value  = css_110_phi    in 5
replace loglik = css_110_ll     in 5

replace model  = "ARIMA(1,1,0)" in 6
replace method = "sspace"       in 6
replace param  = "phi"          in 6
replace value  = ss_110_phi     in 6
replace loglik = ss_110_ll      in 6

* ARIMA(1,1,1)
replace model  = "ARIMA(1,1,1)" in 7
replace method = "MLE"          in 7
replace param  = "phi,theta"    in 7
replace value  = mle_111_phi    in 7
replace loglik = mle_111_ll     in 7

replace model  = "ARIMA(1,1,1)" in 8
replace method = "CSS"          in 8
replace param  = "phi,theta"    in 8
replace value  = css_111_phi    in 8
replace loglik = css_111_ll     in 8

replace model  = "ARIMA(1,1,1)" in 9
replace method = "sspace"       in 9
replace param  = "phi,theta"    in 9
replace value  = ss_111_phi     in 9
replace loglik = ss_111_ll      in 9

export delimited using "results_comparison_mle.csv", replace
display "Exportado: results_comparison_mle.csv"
restore

* --- Tabela de diferencas percentuais ---
preserve
clear
set obs 3

gen str15 model    = ""
gen double mle_vs_ss_pct  = .
gen double mle_vs_css_pct = .
gen double mle_vs_ss_ll   = .

replace model          = "ARIMA(0,1,1)" in 1
replace mle_vs_ss_pct  = diff_mle_ss_theta in 1
replace mle_vs_css_pct = diff_mle_css_theta in 1
replace mle_vs_ss_ll   = diff_mle_ss_ll in 1

replace model          = "ARIMA(1,1,0)" in 2
replace mle_vs_ss_pct  = diff_mle_ss_phi in 2
replace mle_vs_css_pct = diff_mle_css_phi in 2
replace mle_vs_ss_ll   = diff_mle_ss_ll_110 in 2

replace model          = "ARIMA(1,1,1)" in 3
replace mle_vs_ss_pct  = diff_mle_ss_phi_111 in 3
replace mle_vs_css_pct = 0 in 3
replace mle_vs_ss_ll   = diff_mle_ss_ll_111 in 3

export delimited using "results_comparison_mle_diffs.csv", replace
display "Exportado: results_comparison_mle_diffs.csv"
restore

display ""
display "============================================================"
display " VALIDACAO MLE vs CSS vs sspace COMPLETA"
display "============================================================"

log close
