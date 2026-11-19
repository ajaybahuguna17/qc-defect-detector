package com.team.defectqc.advisor

data class DefectReference(
    val description: String,
    val possibleCauses: List<String>,
    val correctiveActions: List<String>,
    val preventiveMeasures: List<String>
)

/** Initial notes for six NEU-DET classes. Validate against cited industrial references. */
object DefectKnowledgeBase {
    private val entries = mapOf(
        "scratches" to DefectReference(
            "Linear marks or grooves on the steel surface.",
            listOf("Abrasive contact with rollers, guides, or handling equipment.",
                "Hard particles trapped between the material and contact surfaces."),
            listOf("Inspect roller and guide surfaces for wear or embedded particles.",
                "Assess scratch depth; recondition only when the approved procedure permits."),
            listOf("Maintain rollers and guides and remove abrasive contamination.",
                "Review material handling and contact points.")
        ),
        "crazing" to DefectReference(
            "A network of fine, interconnected surface cracks.",
            listOf("Thermal or mechanical stresses during processing.",
                "Non-uniform cooling or deformation conditions."),
            listOf("Inspect crack extent and depth using the approved procedure.",
                "Escalate extensive or deep cracking for rejection review."),
            listOf("Review rolling temperature and cooling uniformity.",
                "Check for abnormal thermal or mechanical stress.")
        ),
        "inclusion" to DefectReference(
            "A non-metallic inclusion or inclusion-related surface indication.",
            listOf("Non-metallic particles such as oxides or slag entrained during steelmaking.",
                "Insufficient control of melt cleanliness or slag carry-over."),
            listOf("Assess the indication and material specification before disposition.",
                "Review steelmaking cleanliness and slag-control records."),
            listOf("Strengthen melt-cleanliness and slag-control practices.",
                "Review upstream process records when inclusions recur.")
        ),
        "patches" to DefectReference(
            "Irregular patch-like regions with a different surface appearance.",
            listOf("Non-uniform scale formation or removal.",
                "Local process or surface-treatment variation."),
            listOf("Inspect the affected region and verify whether the appearance is superficial.",
                "Review descaling and surface-treatment conditions."),
            listOf("Maintain consistent descaling conditions.",
                "Monitor process uniformity across the strip or plate.")
        ),
        "pitted_surface" to DefectReference(
            "Small cavities or pits distributed over the steel surface.",
            listOf("Localized corrosion or chemical attack.",
                "Surface contamination or scale-related processing conditions."),
            listOf("Measure pit depth and distribution using the approved inspection method.",
                "Assess whether the material meets its surface specification."),
            listOf("Control storage and exposure conditions to limit corrosion.",
                "Review cleaning, descaling, and surface-protection processes.")
        ),
        "rolled-in_scale" to DefectReference(
            "Oxide scale pressed into the steel surface during rolling.",
            listOf("Scale not fully removed before rolling.",
                "Descaling or rolling conditions that allow oxide to become embedded."),
            listOf("Inspect the area and determine whether approved surface conditioning is possible.",
                "Review descaling effectiveness and rolling conditions."),
            listOf("Maintain descaling equipment and verify scale removal.",
                "Monitor furnace, transfer, and rolling conditions.")
        )
    )

    fun get(type: String): DefectReference? = entries[normalize(type)]
    fun all(): Map<String, DefectReference> = entries

    fun normalize(type: String): String = type.trim().lowercase()
        .replace(' ', '_').replace('-', '_')
        .replace("rolled_in_scale", "rolled-in_scale")
}
