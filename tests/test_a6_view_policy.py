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
