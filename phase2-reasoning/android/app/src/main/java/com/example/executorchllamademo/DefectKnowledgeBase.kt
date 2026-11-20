
package com.example.executorchllamademo

// Stores information about each steel defect
data class DefectInfo(
    val name: String,
    val description: String,
    val possibleCauses: List<String>,
    val correctiveActions: List<String>,
    val preventiveMeasures: List<String>
)

// Knowledge base containing all six NEU-DET defects
object DefectKnowledgeBase {

    private val defects = mapOf(

        "crazing" to DefectInfo(
            name = "Crazing",
            description = "A network of fine, irregular cracks on the steel surface.",
            possibleCauses = listOf(
                "Thermal stress",
                "Surface stress during processing",
                "Improper rolling or cooling conditions"
            ),
            correctiveActions = listOf(
                "Inspect the affected surface",
                "Evaluate crack depth and extent",
                "Segregate severely affected material for further inspection"
            ),
            preventiveMeasures = listOf(
                "Control rolling temperature",
                "Maintain proper cooling conditions",
                "Monitor surface quality during production"
            )
        ),

        "inclusion" to DefectInfo(
            name = "Inclusion",
            description = "Foreign particles or non-metallic impurities embedded in or attached to the steel surface.",
            possibleCauses = listOf(
                "Non-metallic impurities",
                "Slag or oxide contamination",
                "Poor steel cleanliness"
            ),
            correctiveActions = listOf(
                "Inspect the size and distribution of inclusions",
                "Evaluate material quality",
                "Consider surface reconditioning if permitted"
            ),
            preventiveMeasures = listOf(
                "Improve steel cleanliness",
                "Control slag contamination",
                "Monitor the steelmaking process"
            )
        ),

        "patches" to DefectInfo(
            name = "Patches",
            description = "Irregular localized patch-like regions on the steel surface.",
            possibleCauses = listOf(
                "Surface contamination",
                "Uneven processing conditions",
                "Oxide or scale irregularities"
            ),
            correctiveActions = listOf(
                "Inspect the affected region",
                "Evaluate the extent of surface damage",
                "Apply approved surface cleaning if appropriate"
            ),
            preventiveMeasures = listOf(
                "Maintain clean processing surfaces",
                "Monitor rolling conditions",
                "Ensure uniform surface treatment"
            )
        ),

        "pitted_surface" to DefectInfo(
            name = "Pitted Surface",
            description = "Small cavities or depressions appearing on the steel surface.",
            possibleCauses = listOf(
                "Corrosion or chemical attack",
                "Surface damage",
                "Processing irregularities"
            ),
            correctiveActions = listOf(
                "Inspect pit depth and distribution",
                "Evaluate the affected material",
                "Determine whether surface reconditioning is possible"
            ),
            preventiveMeasures = listOf(
                "Control oxidation",
                "Maintain proper storage conditions",
                "Monitor surface processing"
            )
        ),

        "rolled-in_scale" to DefectInfo(
            name = "Rolled-in Scale",
            description = "Oxide scale pressed into the steel surface during the rolling process.",
            possibleCauses = listOf(
                "Inadequate descaling",
                "Oxide formation during heating",
                "Residual scale during rolling"
            ),
            correctiveActions = listOf(
                "Inspect scale penetration",
                "Evaluate the affected surface",
                "Consider approved descaling or reconditioning"
            ),
            preventiveMeasures = listOf(
                "Improve descaling operations",
                "Control furnace conditions",
                "Remove residual scale before rolling"
            )
        ),

        "scratches" to DefectInfo(
            name = "Scratches",
            description = "Linear marks or grooves on the steel surface.",
            possibleCauses = listOf(
                "Mechanical contact",
                "Abrasive particles",
                "Damaged rollers or improper handling"
            ),
            correctiveActions = listOf(
                "Inspect scratch length and depth",
                "Check the affected surface",
                "Evaluate whether reconditioning is possible"
            ),
            preventiveMeasures = listOf(
                "Inspect rollers regularly",
                "Remove abrasive contaminants",
                "Improve material handling"
            )
        )
    )

    // Get information about a particular defect
    fun getDefect(defectName: String): DefectInfo? {
        return defects[defectName.trim().lowercase().replace(" ", "_")]
    }

    // Get information about all defects
    fun getAllDefects(): List<DefectInfo> {
        return defects.values.toList()
    }
}
