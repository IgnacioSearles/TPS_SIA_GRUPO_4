"""Reutilización auditada tras agregar una opción de arquitectura desactivada."""

from src.data.art_capacity import read_json


def audited_legacy_cache(root, name, previous, current):
    """Solo admite los hashes exactos verificados en la prueba de compatibilidad.

    Mantiene inmutables datos, pesos, configuración y métricas. No invalida el
    control de caché general ni permite cambios futuros de código sin revisión.
    """
    proof_path = root/'results/art_search/refinement_compatibility.json'
    if not proof_path.exists():
        return False
    proof = read_json(proof_path)
    field = 'evaluation_sha256'
    if {k:v for k,v in previous.items() if k != field} != {k:v for k,v in current.items() if k != field}:
        return False
    if current.get(field) != proof.get('current_evaluation_sha256'):
        return False
    if current.get('loss_guard') != proof.get('loss_guard'):
        return False
    for item in proof.get('legacy_predictions', []):
        if (item['candidate'] == name and item.get('decoded_outputs_bitwise_identical')
                and item.get('previous_evaluation_sha256') == previous.get(field)):
            stable = ('src/evaluation/art_geometry.py', 'src/evaluation/art_cleanliness.py')
            return all(previous[field][p] == current[field][p] for p in stable)
    return False
