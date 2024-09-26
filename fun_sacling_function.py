# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 18:43:40 2018

@author: nitr
"""

def scaling_fun():
    # prepare Scaling models
    ScalingMethods = []

    # =====================================
    # Scaling feature @ WithOut Scaling'''
    # ======================================
    ScalingMethods.append(('WithoutScaling', ''))

# =============================================================================
#     # =====================================
#     # Scaling feature @ StandardScaler'''
#     # ======================================
#    from sklearn.preprocessing import StandardScaler
#     #sc=StandardScaler()
#     #My1X=sc.fit_transform(MainMyX)
#    ScalingMethods.append(('StandardScaler', StandardScaler()))
#     
#     #===================================
#     #Scaling feature @ MinMaxScaler'''
#     #==================================
#    from sklearn.preprocessing import MinMaxScaler
##     #sc=MinMaxScaler()
##     #My2X=sc.fit_transform(MainMyX)
#    ScalingMethods.append(('MinMaxScaler', MinMaxScaler()))
#     
#     
#     # ==================================
#     #Scaling feature @ RobustScaler'''
#     # ==================================
#     from sklearn.preprocessing import RobustScaler
#     #sc=RobustScaler()
#     #My3X=sc.fit_transform(MainMyX)  
#     ScalingMethods.append(('RobustScaler', RobustScaler()))
#     
# 
#     #====================================
#     #Scaling feature @ Normalizer'''
#     #=================================
#     from sklearn.preprocessing import Normalizer
#     #sc=Normalizer()
#     #My4X=sc.fit_transform(MainMyX)
#     ScalingMethods.append(('Normalizer', Normalizer()))
# 
# 
#     #==========================
#     #Scaling feature @ MaxAbsScaler'''
#     #===============================
#     from sklearn.preprocessing import MaxAbsScaler
#     #sc=MaxAbsScaler()
#     #My5X=sc.fit_transform(MainMyX)
#     ScalingMethods.append(('MaxAbsScaler', MaxAbsScaler()))
#     
# 
#     #=========================
#     #Scaling feature @ QuantileTransformer'''
#     #============================
#     from sklearn.preprocessing import QuantileTransformer
#     #sc=QuantileTransformer()
#     #My6X=sc.fit_transform(MainMyX)
#     ScalingMethods.append(('QuantileTransformer', QuantileTransformer()))
# =============================================================================
    
    return ScalingMethods
