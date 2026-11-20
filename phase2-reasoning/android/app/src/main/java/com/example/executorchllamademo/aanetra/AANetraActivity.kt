package com.example.executorchllamademo.aanetra

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.example.executorchllamademo.WelcomeActivity
import java.io.File

/** Launcher screen: operator inspection UI. LlamaDemo screens are behind the gear icon. */
class AANetraActivity : ComponentActivity() {

    private val viewModel: InspectionViewModel by viewModels()

    private var pendingPhotoUri: Uri? = null

    private val pickImage = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> uri?.let(viewModel::inspect) }

    private val takePicture = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { saved ->
        val uri = pendingPhotoUri
        if (saved && uri != null) viewModel.inspect(uri)
    }

    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) launchCamera()
        else Toast.makeText(this, "Camera permission is needed to take a photo.", Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        pendingPhotoUri = savedInstanceState?.getString(KEY_PHOTO)?.let(Uri::parse)

        setContent {
            AANetraTheme {
                val visionStatus by viewModel.visionStatus.collectAsState()
                val modelState by viewModel.modelState.collectAsState()
                val busy by viewModel.busy.collectAsState()
                val result by viewModel.result.collectAsState()
                val error by viewModel.error.collectAsState()
                val advice by viewModel.advice.collectAsState()

                AANetraScreen(
                    visionStatus = visionStatus,
                    modelState = modelState,
                    busy = busy,
                    result = result,
                    error = error,
                    advice = advice,
                    onPickImage = {
                        pickImage.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                        )
                    },
                    onTakePhoto = ::onTakePhoto,
                    onRetryModel = viewModel::retryModel,
                    onRetryAdvice = viewModel::retryAdvice,
                    onAcceptMobileData = viewModel::acceptMobileData,
                    onDeclineMobileData = viewModel::declineMobileData,
                    onOpenDeveloperSettings = {
                        startActivity(Intent(this, WelcomeActivity::class.java))
                    }
                )
            }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        pendingPhotoUri?.let { outState.putString(KEY_PHOTO, it.toString()) }
    }

    private fun onTakePhoto() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            launchCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun launchCamera() {
        val dir = File(cacheDir, "camera").apply { mkdirs() }
        // Keep only the latest photo in the cache.
        dir.listFiles()?.forEach { it.delete() }
        val file = File(dir, "inspect_${System.currentTimeMillis()}.jpg")
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        pendingPhotoUri = uri
        try {
            takePicture.launch(uri)
        } catch (e: Exception) {
            Toast.makeText(this, "No camera app available.", Toast.LENGTH_LONG).show()
        }
    }

    companion object {
        private const val KEY_PHOTO = "pending_photo_uri"
    }
}
