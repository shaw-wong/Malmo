package net.minecraft.realms;

import net.minecraft.util.ChatAllowedCharacters;
import net.minecraftforge.fml.relauncher.Side;
import net.minecraftforge.fml.relauncher.SideOnly;

@SideOnly(Side.CLIENT)
public class RealmsSharedConstants
{
    public static int NETWORK_PROTOCOL_VERSION = 316;
    public static int TICKS_PER_SECOND = 20;
    public static String VERSION_STRING = "1.11.2";
    public static char[] ILLEGAL_FILE_CHARACTERS = ChatAllowedCharacters.ILLEGAL_FILE_CHARACTERS;
}