// This function is appended to HexGameUI.cpp
// It should be added at the end of the HexGameUI::run() method's closing brace
// Updates volume icon sprites based on slider values.

void HexGameUI::updateVolumeIcon(int sliderIndex, float value) {
    // Determine which icon to show based on volume value
    // vol0.png: 0-25%, vol1.png: 26-50%, vol2.png: 51-75%, vol3.png: 76-100%
    int iconIndex = 3;  // Default to vol3
    
    if (value <= 25.0f) {
        iconIndex = 0;
    } else if (value <= 50.0f) {
        iconIndex = 1;
    } else if (value <= 75.0f) {
        iconIndex = 2;
    } else {
        iconIndex = 3;
    }
    
    if (sliderIndex >= 0 && sliderIndex < 3) {
        volumeIconSprites_[sliderIndex].setTexture(volumeIconTextures_[iconIndex]);
        
        // Update stored icon indices for reference
        if (sliderIndex == 0) {
            masterVolumeIcon_ = iconIndex;
        } else if (sliderIndex == 1) {
            musicVolumeIcon_ = iconIndex;
        } else if (sliderIndex == 2) {
            sfxVolumeIcon_ = iconIndex;
        }
    }
}
