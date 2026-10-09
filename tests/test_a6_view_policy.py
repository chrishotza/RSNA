from src.a6_view_policy import (
    A6_POLICIES, TARGET_POLICY, acquisition_score,
    deterministic_weighted_centers, policy_for_target,
    target_acquisition_matrix,
)

def test_all_policies_valid():
    for p in A6_POLICIES.values():
        p.validate()

def test_pf_prefers_axial():
    p=policy_for_target("PF OA")
    ax=acquisition_score(plane="AXIAL",fluid_sensitive=False,fat_suppression=False,policy=p)
    sag=acquisition_score(plane="SAGITTAL",fluid_sensitive=False,fat_suppression=False,policy=p)
    assert ax > sag

def test_synovitis_prefers_fluid_sensitive():
    p=policy_for_target("Synovitis")
    a=acquisition_score(plane="AXIAL",fluid_sensitive=True,fat_suppression=True,policy=p)
    b=acquisition_score(plane="AXIAL",fluid_sensitive=False,fat_suppression=False,policy=p)
    assert a > b

def test_lateral_prefers_sag_cor_over_axial():
    p=policy_for_target("Lateral Meniscus")
    cor=acquisition_score(plane="CORONAL",fluid_sensitive=False,fat_suppression=False,policy=p)
    ax=acquisition_score(plane="AXIAL",fluid_sensitive=False,fat_suppression=False,policy=p)
    assert cor > ax

def test_centers_deterministic_and_in_bounds():
    p=A6_POLICIES["pf_axial_dense"]
    a=deterministic_weighted_centers(37,p)
    b=deterministic_weighted_centers(37,p)
    assert a==b
    assert a==sorted(set(a))
    assert min(a)>=0 and max(a)<37

def test_matrix_target_specific():
    rows=[
      {"Anatomical_Plane":"AXIAL","Fluid_Sensitive":1,"Fat_Suppression":1},
      {"Anatomical_Plane":"CORONAL","Fluid_Sensitive":0,"Fat_Suppression":0},
    ]
    m=target_acquisition_matrix(rows)
    assert m["PF OA"][0] > m["PF OA"][1]
    assert m["Lateral OA"][1] > 0


def test_attention_prior_shape_and_neutral_strength():
    import numpy as np
    from src.a6_view_policy import attention_log_prior_from_token_types
    tt=np.array([[12,8,4,0],[12,12,8,4]])
    p=attention_log_prior_from_token_types(tt,strength=.35)
    assert p.shape==(2,12,4)
    z=attention_log_prior_from_token_types(tt,strength=0.0)
    assert np.allclose(z,0)

def test_attention_prior_pf_prefers_axial():
    import numpy as np
    from src.a6_view_policy import attention_log_prior_from_token_types, TARGETS
    # axial plain=12, sagittal plain=4
    p=attention_log_prior_from_token_types(np.array([[12,4]]),strength=.35)
    q=TARGETS.index("PF OA")
    assert p[0,q,0] > p[0,q,1]

def test_attention_prior_synovitis_prefers_fluid_fat():
    import numpy as np
    from src.a6_view_policy import attention_log_prior_from_token_types, TARGETS
    # axial fluid+fat=15 vs axial plain=12
    p=attention_log_prior_from_token_types(np.array([[15,12]]),strength=.35)
    q=TARGETS.index("Synovitis")
    assert p[0,q,0] > p[0,q,1]
